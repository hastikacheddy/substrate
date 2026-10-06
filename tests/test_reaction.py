import numpy as np
import pytest

from substrate import ClassicalBackend, Quantity, Scale, ScientificSystem
from substrate.engines.reaction import MassActionEngine


def network(species, reactions, c0, rates, t_end, n_time=300):
    params = {k: Quantity(v, "1/s") for k, v in rates.items()}
    params["t_end"] = Quantity(t_end, "s")
    params["n_time"] = Quantity(n_time, "1")
    return ScientificSystem(
        "net", Scale.REACTION, "reaction.network",
        parameters=params, state={"c": Quantity(c0, "M")},
        structure={"species": species, "reactions": reactions},
    )


def solve(system):
    return MassActionEngine().solve(system, ClassicalBackend())


def isomerisation(kf=3.0, kr=1.0, t_end=5.0):
    return network(
        ["A", "B"],
        [{"id": "f", "reactants": {"A": 1}, "products": {"B": 1}, "rate_parameter": "k_f"},
         {"id": "r", "reactants": {"B": 1}, "products": {"A": 1}, "rate_parameter": "k_r"}],
        [1.0, 0.0], {"k_f": kf, "k_r": kr}, t_end,
    )


def test_reversible_first_order_matches_analytic_solution():
    kf, kr = 3.0, 1.0
    out = solve(isomerisation(kf, kr))
    t = out.obs("time")
    a_eq = kr / (kf + kr)
    exact_a = a_eq + (1 - a_eq) * np.exp(-(kf + kr) * t)
    assert out.obs("concentrations")[:, 0] == pytest.approx(exact_a, abs=1e-7)


def test_total_population_is_conserved():
    c = solve(isomerisation()).obs("concentrations")
    assert c.sum(axis=1) == pytest.approx(np.ones(len(c)), abs=1e-9)


def test_relaxation_time_is_inverse_sum_of_rates():
    out = solve(isomerisation(3.0, 1.0))
    assert out.obs("relaxation_time") == pytest.approx(1 / 4.0, rel=1e-5)


def test_second_order_stoichiometry():
    # 2A -> B with v = k [A]^2 and dA/dt = -2 v  =>  A(t) = A0 / (1 + 2 k A0 t)
    k, a0 = 0.7, 1.5
    system = network(
        ["A", "B"],
        [{"id": "dim", "reactants": {"A": 2}, "products": {"B": 1}, "rate_parameter": "k"}],
        [a0, 0.0], {"k": k}, 4.0,
    )
    out = solve(system)
    t = out.obs("time")
    assert out.obs("concentrations")[:, 0] == pytest.approx(a0 / (1 + 2 * k * a0 * t), rel=1e-6)
    # two A consumed per B formed
    c = out.obs("concentrations")
    assert c[:, 0] + 2 * c[:, 1] == pytest.approx(np.full(len(c), a0), abs=1e-8)
