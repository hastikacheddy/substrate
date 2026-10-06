"""Reaction scale: mass-action chemical kinetics.

A reaction system's structure is

    {"species": ["A", "B"],
     "reactions": [{"id": "A->B", "reactants": {"A": 1}, "products": {"B": 1}, "rate_parameter": "k_f"}, ...]}

with one rate-constant parameter per reaction and the initial concentrations in state["c"].
"""
from __future__ import annotations

import numpy as np

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem

KIND = "reaction.network"


class MassActionEngine(Engine):
    name = "reaction.mass_action_ode"
    scale = Scale.REACTION
    kinds = (KIND,)
    approximations = (
        "well-mixed, deterministic mass-action kinetics (no spatial structure, no stochasticity)",
        "rate constants are constant in time (fixed temperature)",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        species = system.structure["species"]
        reactions = system.structure["reactions"]
        index = {s: i for i, s in enumerate(species)}
        n_s, n_r = len(species), len(reactions)

        stoich = np.zeros((n_s, n_r))      # net change of each species per reaction event
        order = np.zeros((n_r, n_s))       # mass-action exponents
        for j, r in enumerate(reactions):
            for s, nu in r["reactants"].items():
                stoich[index[s], j] -= nu
                order[j, index[s]] += nu
            for s, nu in r["products"].items():
                stoich[index[s], j] += nu
        k = np.array([system.param(r["rate_parameter"]) for r in reactions], dtype=float)
        if np.any(k < 0):
            raise ValidationError(f"{system.name}: negative rate constant")

        def rhs(_t, c):
            return stoich @ (k * np.prod(np.maximum(c, 0.0) ** order, axis=1))

        c0 = np.asarray(system.state["c"].value, dtype=float)
        t_end = system.param("t_end", "s")
        n_time = int(system.param("n_time", default=200))
        t = np.linspace(0.0, t_end, n_time)
        c_t = backend.integrate_ode(rhs, c0, t)

        c_final = c_t[-1]
        jac = _numerical_jacobian(rhs, c_final)
        lam = np.abs(np.linalg.eigvals(jac).real)
        slow = lam[lam > 1e-9 * max(lam.max(), 1e-300)]
        src = self.name
        obs = {
            "time": Quantity(t, "s", source=src),
            "concentrations": Quantity(c_t, system.state["c"].unit, source=src),
            "final_concentrations": Quantity(c_final, system.state["c"].unit, source=src),
        }
        if slow.size:
            obs["relaxation_time"] = Quantity(1.0 / slow.min(), "s", source=src)
        return system.evolve(
            observables=obs,
            state={"c": Quantity(c_final, system.state["c"].unit, source=src)},
            dynamics="dc/dt = S . v(c),  v_j = k_j prod_s c_s^order_js",
        )


def _numerical_jacobian(rhs, c: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    f0 = rhs(0.0, c)
    jac = np.empty((len(c), len(c)))
    for j in range(len(c)):
        step = eps * max(abs(c[j]), 1.0)
        dc = c.copy()
        dc[j] += step
        jac[:, j] = (rhs(0.0, dc) - f0) / step
    return jac
