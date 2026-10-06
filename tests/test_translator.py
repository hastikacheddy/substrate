import numpy as np
import pytest

from substrate import ValidationError
from substrate.units import KB_EV

from conftest import DEUTERON, double_well


def rates(pipeline, **kw):
    r = pipeline.run(double_well(**kw), ["reaction"])
    return r.final.param("k_f"), r.final.param("k_r"), r


def test_detailed_balance_equilibrium_is_ratio_of_well_partition_functions(pipeline):
    k_f, k_r, r = rates(pipeline)
    quantum = r.trace[0]
    kt = KB_EV * 300.0
    left, right = quantum.obs("levels_left"), quantum.obs("levels_right")
    expected = np.exp(-(right - left[0]) / kt).sum() / np.exp(-(left - left[0]) / kt).sum()
    assert k_f / k_r == pytest.approx(expected, rel=1e-9)
    # the kinetics run stops at 10 relaxation times, so ~e^-10 of the deviation from equilibrium remains
    assert r.final.obs("final_concentrations")[1] == pytest.approx(expected / (1 + expected), rel=2e-4)


def test_rate_constant_equals_level_sum_formula(pipeline):
    k_f, _, r = rates(pipeline)
    q = r.trace[0]
    kt = KB_EV * 300.0
    left = q.obs("levels_left")
    w = np.exp(-(left - left[0]) / kt)
    manual = (q.obs("attempt_frequency_left") * q.obs("transmission_left") * w).sum() / w.sum()
    assert k_f == pytest.approx(manual, rel=1e-12)


def test_rate_rises_with_temperature_and_flattens_into_a_tunnelling_plateau(pipeline):
    k = {t: rates(pipeline, temperature=t)[0] for t in (100, 150, 200, 300, 400)}
    ordered = [k[t] for t in (100, 150, 200, 300, 400)]
    assert all(b >= a for a, b in zip(ordered, ordered[1:]))
    assert k[150] / k[100] < 1.01          # plateau: ground-state tunnelling, no thermal activation
    assert k[400] / k[300] > 1.5           # activated regime


def test_kinetic_isotope_effect_is_large_and_grows_on_cooling(pipeline):
    kie = {t: rates(pipeline, temperature=t)[0] / rates(pipeline, temperature=t, mass=DEUTERON)[0]
           for t in (200, 300, 400)}
    assert kie[400] > 1
    assert kie[200] > kie[300] > kie[400]


def test_deeper_barrier_slows_the_reaction(pipeline):
    assert rates(pipeline, barrier=0.6)[0] < rates(pipeline, barrier=0.5)[0] < rates(pipeline, barrier=0.4)[0]


def test_barrier_below_zero_point_energy_blocks_translation(pipeline):
    with pytest.raises(ValidationError, match="zero-point") as exc:
        pipeline.run(double_well(barrier=0.05), ["reaction"])
    assert exc.value.issues and exc.value.issues[0].severity == "error"


def test_marginal_barrier_translates_but_warns(pipeline):
    r = pipeline.run(double_well(barrier=0.2, temperature=300), ["reaction"])
    assert any("above-barrier" in w for w in r.warnings())


def test_translator_requires_temperature_and_a_solved_source(pipeline):
    system = double_well()
    del system.parameters["temperature"]
    with pytest.raises(ValidationError, match="temperature"):
        pipeline.run(system, ["reaction"])

    from substrate.translators.quantum_to_reaction import QuantumToReaction
    unsolved = QuantumToReaction().validate(double_well())
    assert unsolved and "not been solved" in unsolved[0].message


def test_derivation_records_which_level_carries_the_flux(pipeline):
    _, _, r = rates(pipeline)
    d = r.trace[1].structure["derivation"]
    assert d["dominant_level"] == 0                      # 300 K: ground-state tunnelling dominates
    assert sum(d["flux_fraction_by_level"]) == pytest.approx(1.0)
    _, _, hot = rates(pipeline, temperature=500)
    assert hot.trace[1].structure["derivation"]["flux_fraction_by_level"][1] > d["flux_fraction_by_level"][1]
