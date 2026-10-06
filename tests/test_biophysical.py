import numpy as np
import pytest

from substrate import (
    AmbiguousPathError, ClassicalBackend, Quantity, Scale, ScientificSystem, SubstrateError, ValidationError,
)
from substrate.engines.biophysical import EnzymeCycleEngine, enzyme_cycle_structure
from substrate.engines.reaction import MassActionEngine
from substrate.translators.reaction_to_enzyme import ReactionToEnzyme
from substrate.units import H_EV_S, KB_EV

from conftest import DEUTERON, EXPERIMENTS, enzyme, evb, evb2d, with_context

BACKEND = ClassicalBackend()
ROUTE_M = ["electronic_structure", "molecular", "reaction", "biophysical"]
ROUTE_Q = ["electronic_structure", "quantum", "reaction", "biophysical"]


def solve(**kw):
    return EnzymeCycleEngine().solve(enzyme(**kw), BACKEND)


def analytic_mm(k1, km1, k2, km2, k3):
    """Textbook kcat and K_M for E + S <-> ES <-> EP -> E + P (derived by hand from the steady-state balances)."""
    kcat = k2 * k3 / (k2 + km2 + k3)
    km = (km1 * km2 + km1 * k3 + k2 * k3) / (k1 * (k2 + km2 + k3))
    return kcat, km


# =====================================================================================================
# the engine, against closed-form and independent calculations
# =====================================================================================================
@pytest.mark.parametrize("k_f,k_r,k_on,k_off,k_rel", [
    (1e5, 1e4, 1e7, 1e3, 1e4),
    (1e2, 5e3, 3e6, 4e2, 7e5),
    (8e6, 5e7, 1e8, 1e5, 1e3),
    (1e9, 1e9, 1e5, 1e1, 1e9),
])
def test_kcat_and_km_match_the_closed_form_michaelis_menten_expressions(k_f, k_r, k_on, k_off, k_rel):
    out = solve(k_f=k_f, k_r=k_r, k_on=k_on, k_off=k_off, k_release=k_rel)
    kcat, km = analytic_mm(k_on, k_off, k_f, k_r, k_rel)
    assert out.obs("kcat") == pytest.approx(kcat, rel=1e-8)
    assert out.obs("KM") == pytest.approx(km, rel=1e-8)
    assert out.obs("kcat_over_KM") == pytest.approx(kcat / km, rel=1e-8)


def test_turnover_is_the_michaelis_menten_hyperbola_and_populations_are_a_distribution():
    for s in (1e-7, 1e-5, 1e-4, 1e-2, 1.0):
        out = solve(substrate=s)
        kcat, km = out.obs("kcat"), out.obs("KM")
        assert out.obs("turnover_rate") == pytest.approx(kcat * s / (km + s), rel=1e-8)
        pops = [out.obs(f"population_{x}") for x in ("E", "ES", "EP")]
        assert sum(pops) == pytest.approx(1.0, abs=1e-12) and min(pops) >= 0
        assert out.obs("fraction_bound") == pytest.approx(1 - pops[0])
    assert solve(substrate=1e-2).obs("turnover_rate") > solve(substrate=1e-5).obs("turnover_rate")
    assert solve(substrate=1e6).obs("turnover_rate") == pytest.approx(solve().obs("kcat"), rel=1e-4)   # saturation


def test_steady_state_agrees_with_an_independent_mass_action_ode_simulation():
    # A second, unrelated code path: integrate the full bimolecular network and measure the turnover directly.
    k_on, k_off, k_f, k_r, k_rel = 1e4, 1e2, 3e3, 1e3, 2e3
    e_total, s0 = 1e-6, 1e-3
    reactions = [
        {"id": "bind", "reactants": {"E": 1, "S": 1}, "products": {"ES": 1}, "rate_parameter": "kon"},
        {"id": "unbind", "reactants": {"ES": 1}, "products": {"E": 1, "S": 1}, "rate_parameter": "koff"},
        {"id": "chem_f", "reactants": {"ES": 1}, "products": {"EP": 1}, "rate_parameter": "kf"},
        {"id": "chem_r", "reactants": {"EP": 1}, "products": {"ES": 1}, "rate_parameter": "kr"},
        {"id": "release", "reactants": {"EP": 1}, "products": {"E": 1, "P": 1}, "rate_parameter": "krel"},
    ]
    net = ScientificSystem(
        "ode", Scale.REACTION, "reaction.network",
        parameters={"kon": Quantity(k_on, "1"), "koff": Quantity(k_off, "1/s"), "kf": Quantity(k_f, "1/s"),
                    "kr": Quantity(k_r, "1/s"), "krel": Quantity(k_rel, "1/s"),
                    "t_end": Quantity(0.4, "s"), "n_time": Quantity(801, "1")},
        state={"c": Quantity([e_total, 0.0, 0.0, s0, 0.0], "M")},
        structure={"species": ["E", "ES", "EP", "S", "P"], "reactions": reactions},
    )
    sim = MassActionEngine().solve(net, BACKEND)
    t, c = sim.obs("time"), sim.obs("concentrations")
    late = t > 0.2                                                   # long after the pre-steady-state transient
    rate_per_enzyme = np.polyfit(t[late], c[late, 4], 1)[0] / e_total
    expected = solve(k_f=k_f, k_r=k_r, k_on=k_on, k_off=k_off, k_release=k_rel, substrate=s0).obs("turnover_rate")
    assert rate_per_enzyme == pytest.approx(expected, rel=5e-3)      # the residual is real: S is depleted by the enzyme


def test_haldane_relation_detailed_balance_gives_zero_net_flux_at_equilibrium():
    out = solve(k_on_product=3e6)
    keq = out.obs("overall_equilibrium_constant")
    assert keq == pytest.approx((1e7 * 1e5 * 1e4) / (1e3 * 1e4 * 3e6), rel=1e-12)
    s = 1e-4
    at_equilibrium = solve(k_on_product=3e6, substrate=s, product=keq * s)
    forward = solve(k_on_product=3e6, substrate=s, product=0.0).obs("turnover_rate")
    assert abs(at_equilibrium.obs("turnover_rate")) < 1e-10 * forward         # no net flux when [P]/[S] = K_eq
    assert solve(k_on_product=3e6, substrate=s, product=2 * keq * s).obs("turnover_rate") < 0   # reverses beyond it
    assert solve(k_on_product=3e6, substrate=s, product=0.5 * keq * s).obs("turnover_rate") > 0


def test_flux_control_coefficients_obey_the_summation_theorem_and_the_closed_form():
    out = solve(k_f=1e5, k_r=1e4, k_release=1e4)
    c = {n: out.obs(f"control_{n}") for n in ("k_on", "k_off", "k_f", "k_r", "k_release")}
    assert sum(c.values()) == pytest.approx(1.0, abs=1e-6)                   # kcat is homogeneous of degree 1 in the rates
    assert c["k_on"] == 0 and c["k_off"] == 0                                # binding does not enter kcat
    total = 1e5 + 1e4 + 1e4
    assert c["k_release"] == pytest.approx((1e5 + 1e4) / total, rel=1e-5)    # d ln kcat / d ln k3
    assert c["k_f"] == pytest.approx((1e4 + 1e4) / total, rel=1e-5)
    assert c["k_r"] == pytest.approx(-1e4 / total, rel=1e-5)


def test_binding_thermodynamics():
    out = solve(k_on=1e7, k_off=1e3, temperature=300.0)
    assert out.obs("K_d") == pytest.approx(1e-4)
    assert out.obs("delta_g_binding") == pytest.approx(KB_EV * 300.0 * np.log(1e-4), rel=1e-12)   # -0.238 eV vs a 1 M standard state
    assert out.obs("chemical_equilibrium_constant") == pytest.approx(10.0)


def test_a_slow_release_hides_the_isotope_effect_and_a_fast_release_reveals_it():
    # H and D chemistry: the forward and reverse rates both fall by 50x, so the *equilibrium* isotope effect is 1.
    def kie(release):
        h = solve(k_f=5e5, k_r=1e5, k_release=release).obs("kcat")
        d = solve(k_f=5e5 / 50, k_r=1e5 / 50, k_release=release).obs("kcat")
        return h / d
    assert kie(1e12) == pytest.approx(50, rel=0.01)       # chemistry limits: the intrinsic effect is fully expressed
    assert kie(1e1) == pytest.approx(1.0, abs=0.01)       # release limits: the chemistry equilibrates and is invisible
    assert kie(1e12) > kie(1e6) > kie(1e4) > kie(1e1)     # commitment masks the effect progressively


def test_a_mechanism_with_substrate_inhibition_is_refused_not_misreported():
    structure = {
        "states": ["E", "ES", "EP", "ESS"],
        "steps": [
            {"id": "bind", "from": "E", "to": "ES", "rate_parameter": "context.k_on", "ligand": "substrate"},
            {"id": "unbind", "from": "ES", "to": "E", "rate_parameter": "context.k_off"},
            {"id": "chem_f", "from": "ES", "to": "EP", "rate_parameter": "k_f"},
            {"id": "chem_r", "from": "EP", "to": "ES", "rate_parameter": "k_r"},
            {"id": "release", "from": "EP", "to": "E", "rate_parameter": "context.k_release"},
            {"id": "bind2", "from": "ES", "to": "ESS", "rate_parameter": "context.k_on2", "ligand": "substrate"},
            {"id": "unbind2", "from": "ESS", "to": "ES", "rate_parameter": "context.k_off2"},
        ],
        "turnover": {"forward": "chem_f", "reverse": "chem_r"}, "binding": {"on": "bind", "off": "unbind"},
    }
    system = enzyme().evolve(structure=structure)
    system.parameters["context.k_on2"] = Quantity(1e7, "1/(M s)")
    system.parameters["context.k_off2"] = Quantity(1e3, "1/s")
    with pytest.raises(ValidationError, match="not a Michaelis-Menten hyperbola"):
        EnzymeCycleEngine().solve(system, BACKEND)


def test_engine_validation():
    engine = EnzymeCycleEngine()
    with pytest.raises(ValidationError, match="negative rate"):
        engine.solve(enzyme(k_f=-1.0), BACKEND)
    wrong = enzyme()
    wrong.parameters["context.k_on"].unit = "1/s"
    with pytest.raises(ValidationError, match="expected '1/\\(M s\\)'"):
        engine.solve(wrong, BACKEND)
    with pytest.raises(ValidationError, match="no net forward turnover"):
        engine.solve(enzyme(k_release=0.0), BACKEND)                       # product can never leave
    with pytest.raises(ValidationError, match="disconnected or absorbing"):
        engine.solve(enzyme(substrate=0.0, k_off=0.0, k_release=0.0), BACKEND)   # two closed classes: no unique steady state
    with pytest.raises(ValidationError, match="non-negative"):
        engine.solve(enzyme(substrate=-1.0), BACKEND)
    missing = enzyme()
    del missing.parameters["context.substrate"]
    with pytest.raises(ValidationError, match="missing parameter 'context.substrate'"):
        engine.solve(missing, BACKEND)
    bad_state = enzyme()
    bad_state.structure["steps"][0]["to"] = "ESX"
    with pytest.raises(ValidationError, match="unknown state"):
        engine.solve(bad_state, BACKEND)
    with pytest.raises(ValidationError, match="solves biophysical"):
        engine.solve(evb(), BACKEND)


def test_the_steady_state_is_accurate_when_the_chemistry_is_ten_orders_of_magnitude_faster_than_the_rest():
    # Regression: solving Q pi = 0 and taking the net flux across the chemical step lost ~6 digits here (a difference of
    # two ~1e13 numbers), enough to break the Michaelis-Menten consistency check on some ensemble draws.
    from fractions import Fraction as F

    def exact(k_on, k_off, k_f, k_r, k_rel, s):                      # exact rational arithmetic for E <-> ES <-> EP -> E
        a, b, c, d, e = F(k_on) * F(s), F(k_off), F(k_f), F(k_r), F(k_rel)
        p_es = 1 / (1 + (b + c * e / (d + e)) / a + c / (d + e))
        return float(e * c / (d + e) * p_es)

    for k_f, k_r in ((3.164e13, 1.786e14), (1e14, 1e14), (1e9, 1e12)):
        for s in (1e-6, 1e-4, 1e-3):
            got = solve(k_f=k_f, k_r=k_r, k_release=4888.7, substrate=s).obs("turnover_rate")
            assert got == pytest.approx(exact(1e7, 1e3, k_f, k_r, 4888.7, s), rel=1e-12)
    out = solve(k_f=3.164e13, k_r=1.786e14, k_release=4888.7)         # and k_cat / K_M still come out
    assert out.obs("kcat") == pytest.approx(analytic_mm(1e7, 1e3, 3.164e13, 1.786e14, 4888.7)[0], rel=1e-9)


def test_gth_stationary_distribution_matches_linear_algebra_and_rejects_reducible_chains():
    from substrate.engines.biophysical import stationary_distribution
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 8):
        rates = rng.uniform(0.1, 10.0, (n, n))
        np.fill_diagonal(rates, 0.0)
        q = rates.T - np.diag(rates.sum(axis=1))                     # dpi/dt = Q pi
        a = q.copy()
        a[-1, :] = 1.0
        expected = np.linalg.solve(a, np.eye(n)[-1])
        assert stationary_distribution(rates) == pytest.approx(expected, rel=1e-12)
    assert stationary_distribution(np.array([[0.0, 3.0], [1.0, 0.0]])) == pytest.approx([0.25, 0.75])   # pi_i ~ 1/rate out
    with pytest.raises(ValueError, match="reducible"):
        stationary_distribution(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))


# =====================================================================================================
# context parameters
# =====================================================================================================
def test_context_parameters_travel_through_every_scale_unchanged(pipeline):
    root = with_context(evb(), sigmas={"k_release": 500.0}, k_on=1e7, k_off=1e3, k_release=1e4, substrate=1e-4)
    r = pipeline.run(root, ["biophysical"])
    assert len(r.trace) == 7
    for system in r.trace:
        assert system.param("context.k_on", "1/(M s)") == 1e7 and system.param("context.substrate", "M") == 1e-4
        assert system.parameters["context.k_release"].sigma == 500.0
    assert not any(name.startswith("context.") for name in evb().parameters)      # and they were added, not built in


def test_context_parameters_enter_the_ensemble_like_any_input(pipeline):
    root = with_context(evb(), sigmas={"k_release": 2000.0}, k_on=1e7, k_off=1e3, k_release=1e4, substrate=1e-4)
    r = pipeline.run(root, ["biophysical"], n_samples=20, seed=3)
    assert r.ensemble["varied"] == ["context.k_release"] and r.ensemble["n_ok"] == 20
    assert r.final.observables["kcat"].sigma > 0
    assert r.final.observables["K_d"].sigma is None                # K_d does not depend on the varied release rate


# =====================================================================================================
# the reaction -> biophysical translator
# =====================================================================================================
def reaction_system(pipeline, R=2.6):
    """A reaction system with moderate rates (R = 2.6 keeps the chemistry well under the barrierless limit)."""
    return pipeline.run(with_context(evb(R=R)), ["reaction"]).final


def test_translation_makes_the_reaction_the_chemical_step(pipeline):
    reaction = reaction_system(pipeline)                                    # R = 2.6 ...
    out = pipeline.run(with_context(evb(R=2.6)), ["biophysical"])           # ... for both, so they are the same reaction
    handed = out.trace[5]                                                   # the unsolved biophysical system
    assert handed.scale == Scale.BIOPHYSICAL and handed.kind == "biophysical.enzyme_cycle"
    assert handed.param("k_f") == pytest.approx(reaction.param("k_f")) and handed.param("k_r") == pytest.approx(reaction.param("k_r"))
    assert out.final.obs("chemical_equilibrium_constant") == pytest.approx(reaction.param("K_eq"), rel=1e-12)
    assert handed.structure["derivation"]["from"].endswith("[reaction]")


def test_translator_names_every_missing_context_parameter(pipeline):
    with pytest.raises(ValidationError) as exc:
        pipeline.run(evb(), ["biophysical"])
    for name in ("context.k_on", "context.k_off", "context.k_release", "context.substrate"):
        assert name in str(exc.value)


def test_translator_validation_cases(pipeline):
    reaction = reaction_system(pipeline)
    translator = ReactionToEnzyme()
    assert translator.validate(reaction) == []

    negative = reaction.evolve(parameters={**reaction.parameters, "context.k_off": Quantity(-1.0, "1/s")})
    assert "must be positive" in translator.validate(negative)[0].message

    too_fast = reaction.evolve(parameters={**reaction.parameters, "context.k_on": Quantity(1e11, "1/(M s)")})
    assert any("diffusion limit" in i.message for i in translator.validate(too_fast))

    barrierless = reaction.evolve(parameters={**reaction.parameters, "k_f": Quantity(5e12, "1/s")})
    assert any("barrierless" in i.message for i in translator.validate(barrierless))

    not_two_state = reaction.evolve(structure={"species": ["A", "B", "C"], "reactions": []})
    assert "two-state" in translator.validate(not_two_state)[0].message


def test_a_nearly_barrierless_chemical_step_warns_through_the_pipeline(pipeline):
    r = pipeline.run(with_context(evb(R=2.35)), ["biophysical"])         # a short O...O: the chemistry is ~1e13 1/s
    assert any("barrierless" in w for w in r.warnings())


# =====================================================================================================
# the whole chain
# =====================================================================================================
def test_seven_stage_chain_with_unbroken_provenance(pipeline):
    r = pipeline.run(with_context(evb()), ["biophysical"])
    assert [s.scale for s in r.trace] == [
        Scale.ELECTRONIC_STRUCTURE, Scale.QUANTUM, Scale.QUANTUM, Scale.REACTION, Scale.REACTION,
        Scale.BIOPHYSICAL, Scale.BIOPHYSICAL,
    ]
    previous = r.root
    for system in r.trace:
        record = system.provenance[-1]
        assert record.input_fingerprint == previous.fingerprint() and record.output_fingerprint == system.fingerprint()
        previous = system
    assert len(r.final.provenance) == 7


def test_the_2d_electronic_kind_has_two_routes_to_the_biophysical_scale(pipeline):
    with pytest.raises(AmbiguousPathError, match="3-hop"):
        pipeline.plan(with_context(evb2d()), ["biophysical"])
    assert len(pipeline.plan(with_context(evb2d()), ROUTE_M)) == 7
    assert len(pipeline.plan(with_context(evb2d()), ROUTE_Q)) == 7


def test_when_release_limits_turnover_the_two_routes_agree_despite_disagreeing_on_the_chemistry(pipeline):
    m = pipeline.run(with_context(evb2d()), ROUTE_M)
    q = pipeline.run(with_context(evb2d()), ROUTE_Q)
    # the chemical rate constants differ by ~4 orders of magnitude between the routes ...
    assert q.trace[3].param("k_f") > 1e3 * m.trace[3].param("k_f")
    # ... but with a slow release only the chemistry's equilibrium matters, and the two routes agree on it
    assert m.final.obs("kcat") == pytest.approx(q.final.obs("kcat"), rel=0.2)
    assert m.final.obs("control_k_release") == pytest.approx(1.0, abs=1e-3)      # 1 - k_rel / (k_f + k_r + k_rel)


def test_when_chemistry_limits_turnover_the_routes_disagree(pipeline):
    fast_release = dict(k_on=1e7, k_off=1e3, k_release=1e12, substrate=1e-4)
    m = pipeline.run(with_context(evb2d(), **fast_release), ROUTE_M).final.obs("kcat")
    q = pipeline.run(with_context(evb2d(), **fast_release), ROUTE_Q).final.obs("kcat")
    assert q > 1e3 * m


def test_the_intrinsic_isotope_effect_from_the_molecular_scale_is_masked_by_a_slow_release(pipeline):
    def observed(release):
        ctx = dict(k_on=1e7, k_off=1e3, k_release=release, substrate=1e-4)
        k = lambda mass: pipeline.run(with_context(evb2d(mass=mass), **ctx), ROUTE_M)
        h, d = k(1.007276), k(DEUTERON)
        intrinsic = h.trace[3].param("k_f") / d.trace[3].param("k_f")
        return intrinsic, h.final.obs("kcat") / d.final.obs("kcat")
    intrinsic, fast = observed(1e12)
    _, slow = observed(1e3)
    assert 4 < intrinsic < 15                                        # the zero-point isotope effect from the molecular scale
    assert fast == pytest.approx(intrinsic, rel=0.05)                # fast release: kcat reports the chemistry
    assert slow == pytest.approx(1.0, abs=0.1)                       # slow release: kcat reports the release step


def test_kcat_over_km_reports_the_chemistry_only_when_the_substrate_is_not_sticky(pipeline):
    def kie_on_specificity(k_off):
        ctx = dict(k_on=1e10, k_off=k_off, k_release=1e12, substrate=1e-4)
        k = lambda mass: pipeline.run(with_context(evb2d(mass=mass), **ctx), ROUTE_M)
        h, d = k(1.007276), k(DEUTERON)
        return (h.final.obs("kcat_over_KM") / d.final.obs("kcat_over_KM"),
                h.trace[3].param("k_f") / d.trace[3].param("k_f"))
    sticky, intrinsic = kie_on_specificity(1e4)
    free, _ = kie_on_specificity(1e9)
    assert sticky == pytest.approx(1.0, abs=0.03)                    # k_off << k_f: every bound substrate reacts, kcat/KM = k_on
    assert free == pytest.approx(intrinsic, rel=0.02)                # k_off >> k_f: kcat/KM reports the chemistry
    assert sticky < kie_on_specificity(1e6)[0] < kie_on_specificity(1e8)[0] < free


def test_the_enzyme_example_experiment_runs_from_the_cli(capsys):
    from substrate.cli import main
    assert main(["run", str(EXPERIMENTS / "proton_transfer_enzyme.yaml"), "--samples", "4"]) == 0
    out = capsys.readouterr().out
    assert "biophysical.enzyme_cycle" in out and "kcat_over_KM" in out and "delta_g_binding" in out
