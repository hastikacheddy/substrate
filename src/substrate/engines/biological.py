"""Biological scale: a metabolic network at steady state, with metabolic control analysis.

A network is a set of internal metabolites and reactions between them and fixed (boundary) pools. Each reaction has a
rate law drawn from a small library, with its parameters named in the system:

    exchange          k (a - b)                                      transport / diffusion between two pools
    michaelis_menten  vmax x / (km + x)                              an irreversible enzyme
    reversible_mm     (vf s/ks - vr p/kp) / (1 + s/ks + p/kp)        a reversible enzyme, thermodynamically consistent
    first_order       k x                                            degradation, dilution
    constant          v0                                             a fixed influx

A reference to a concentration (`a`, `s`, `x`, ...) names either an internal metabolite (dynamic) or a parameter in M (a
boundary pool held fixed).

The engine finds the steady state by following the system's own dynamics until it settles and then polishing the result, checks
that it is stable, reports concentrations and fluxes, and computes flux-control coefficients: how much each reaction controls
the pathway flux, `C_i = d ln J / d ln a_i` where `a_i` scales reaction i's rate. They sum to 1 (the summation theorem).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem

KIND = "biological.metabolic_network"

#: Right-hand-side evaluations one steady-state search may spend. Healthy networks need ~300-1500 (measured), so this is
#: ~20x headroom over the worst case seen. A network whose product accumulates without bound makes an adaptive integrator
#: take steps for minutes, and no convergence test between integration calls can interrupt a call that never returns.
INTEGRATION_BUDGET = 30_000


#: no solute is present at more than this (M); pure water is ~55 M. A "steady state" above it is the model being run far
#: outside the dilute-solution regime it describes (usually a rate that is wrong by orders of magnitude).
MAX_CONCENTRATION = 100.0


class _BudgetExceeded(Exception):
    pass


def _exchange(x, p):
    return p["k"] * (x["a"] - x["b"])


def _michaelis_menten(x, p):
    return p["vmax"] * x["x"] / (p["km"] + x["x"])


def _reversible_mm(x, p):
    s, q = x["s"] / p["ks"], x["p"] / p["kp"]
    return (p["vf"] * s - p["vr"] * q) / (1.0 + s + q)


def _first_order(x, p):
    return p["k"] * x["x"]


def _constant(x, p):
    return p["v0"]


#: rate laws: the concentration references they take, their parameters with units, and the function
_LAWS = {
    "exchange": (("a", "b"), {"k": "1/s"}, _exchange),
    "michaelis_menten": (("x",), {"vmax": "M/s", "km": "M"}, _michaelis_menten),
    "reversible_mm": (("s", "p"), {"vf": "M/s", "ks": "M", "vr": "M/s", "kp": "M"}, _reversible_mm),
    "first_order": (("x",), {"k": "1/s"}, _first_order),
    "constant": ((), {"v0": "M/s"}, _constant),
}
_MUST_BE_POSITIVE = {"km", "ks", "kp"}          # they appear in denominators


def pathway_structure(reversible: bool) -> dict:
    """external substrate -> S --enzyme--> P -> drain. The enzyme's parameters (`enzyme_*`) come from the biophysical scale;
    everything named `context.*` is an experiment-level input."""
    if reversible:
        enzyme = {"law": "reversible_mm", "refs": {"s": "S", "p": "P"},
                  "params": {"vf": "enzyme_vf", "ks": "enzyme_ks", "vr": "enzyme_vr", "kp": "enzyme_kp"}}
    else:
        enzyme = {"law": "michaelis_menten", "refs": {"x": "S"}, "params": {"vmax": "enzyme_vf", "km": "enzyme_ks"}}
    return {
        "species": ["S", "P"],
        "reactions": [
            {"id": "transport", "law": "exchange", "stoich": {"S": 1},
             "refs": {"a": "context.external_substrate", "b": "S"}, "params": {"k": "context.transport_rate"}},
            {"id": "enzyme", "stoich": {"S": -1, "P": 1}, **enzyme},
            {"id": "drain", "law": "michaelis_menten", "stoich": {"P": -1}, "refs": {"x": "P"},
             "params": {"vmax": "context.drain_vmax", "km": "context.drain_km"}},
        ],
        "flux_reaction": "enzyme",
    }


class _Network:
    def __init__(self, structure: dict, values: dict[str, float], backend: SolverBackend):
        self.species = structure["species"]
        self.reactions = structure["reactions"]
        self.index = {s: i for i, s in enumerate(self.species)}
        self.values = values
        self.backend = backend
        self.stoich = np.zeros((len(self.species), len(self.reactions)))
        for j, r in enumerate(self.reactions):
            for s, coeff in r["stoich"].items():
                self.stoich[self.index[s], j] = coeff
        # A reference rate scale: every reaction evaluated with all pools at the largest boundary concentration. Measuring
        # balance against this (rather than against the current rates alone) keeps "settled" meaningful at exact equilibrium,
        # where every net rate is ~0 and a purely relative test would compare noise with noise.
        boundary = [v for n, v in values.items() if any(n in r["refs"].values() for r in self.reactions) and n not in self.index]
        pool = max([b for b in boundary if b > 0] or [1e-6])
        self.v_ref = max(np.abs(self.rates(np.full(len(self.species), pool), np.ones(len(self.reactions)))).max(), 1e-300)

    def _scale(self, c: np.ndarray, activity: np.ndarray) -> float:
        return max(np.abs(self.rates(c, activity)).max(), 1e-6 * self.v_ref)

    def rates(self, c: np.ndarray, activity: np.ndarray) -> np.ndarray:
        c = np.maximum(c, 0.0)
        v = np.empty(len(self.reactions))
        for j, r in enumerate(self.reactions):
            x = {k: (c[self.index[n]] if n in self.index else self.values[n]) for k, n in r["refs"].items()}
            p = {k: self.values[n] for k, n in r["params"].items()}
            v[j] = activity[j] * _LAWS[r["law"]][2](x, p)
        return v

    def rhs(self, c: np.ndarray, activity: np.ndarray) -> np.ndarray:
        return self.stoich @ self.rates(c, activity)

    def imbalance(self, c: np.ndarray, activity: np.ndarray) -> float:
        """Largest net production rate of any metabolite, relative to the fastest reaction (floored at 1e-6 of the
        reference scale)."""
        return float(np.abs(self.rhs(c, activity)).max() / self._scale(c, activity))

    def steady_state(self, c0: np.ndarray, activity: np.ndarray) -> np.ndarray:
        c = np.maximum(np.asarray(c0, dtype=float), 0.0)
        horizon = 1e-6
        remaining = [INTEGRATION_BUDGET]

        def metered(_t, cc):
            remaining[0] -= 1
            if remaining[0] < 0:
                raise _BudgetExceeded
            return self.rhs(cc, activity)

        try:
            for _ in range(18):                              # follow the dynamics, 10x longer each time, until settled
                if self.imbalance(c, activity) < 1e-8:
                    break
                y = self.backend.integrate_ode(metered, c, np.array([0.0, horizon]))
                c = np.maximum(y[-1], 0.0)
                horizon *= 10.0
            else:
                raise ValidationError("no steady state reached: the network keeps changing (unbounded growth or oscillation)")
        except _BudgetExceeded:
            raise ValidationError(
                "no steady state reached within the integration budget: the dynamics are extremely stiff or some "
                "metabolite grows without bound (a production rate far above anything that consumes it?)"
            ) from None
        except RuntimeError as error:                        # the backend's integrator gave up
            raise ValidationError(f"no steady state reached: the integrator failed ({error})") from None
        scale_c = np.maximum(c, 1e-12)                        # polish with Newton-type root finding on scaled variables
        scale_v = self._scale(c, activity)
        sol = root(lambda x: self.rhs(x * scale_c, activity) / scale_v, np.ones(len(c)), method="hybr", tol=1e-14)
        polished = np.maximum(sol.x * scale_c, 0.0)
        return polished if sol.success and self.imbalance(polished, activity) <= self.imbalance(c, activity) else c

    def jacobian(self, c: np.ndarray, activity: np.ndarray) -> np.ndarray:
        n = len(c)
        jac = np.empty((n, n))
        for j in range(n):
            h = 1e-6 * max(c[j], 1e-12)
            up, down = c.copy(), c.copy()
            up[j] += h
            down[j] = max(down[j] - h, 0.0)
            jac[:, j] = (self.rhs(up, activity) - self.rhs(down, activity)) / (up[j] - down[j])
        return jac


class MetabolicNetworkEngine(Engine):
    name = KIND
    scale = Scale.BIOLOGICAL
    kinds = (KIND,)
    approximations = (
        "a well-mixed compartment: no spatial structure, no stochasticity, deterministic mass-action-level kinetics",
        "enzymes follow quasi-steady-state rate laws (enzyme concentration well below K_M + [S]), with constant total amounts",
        "no regulation, gene expression, growth or cell division: parameters do not change in time",
        "the steady state reached from the initial concentrations is the one analysed; others, if any, are not explored",
        "boundary pools are held fixed regardless of what the network does to them",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        structure = system.structure
        species = structure["species"]
        reactions = structure["reactions"]
        ids = [r["id"] for r in reactions]
        if len(set(ids)) != len(ids) or structure.get("flux_reaction") not in ids:
            raise ValidationError(f"{system.name}: reaction ids must be unique and flux_reaction must name one of them")

        values: dict[str, float] = {}
        for r in reactions:
            if r["law"] not in _LAWS:
                raise ValidationError(f"{system.name}: unknown rate law '{r['law']}' (have: {', '.join(_LAWS)})")
            refs, params, _ = _LAWS[r["law"]]
            if set(r["refs"]) != set(refs) or set(r["params"]) != set(params):
                raise ValidationError(f"{system.name}: reaction '{r['id']}' needs refs {list(refs)} and params {list(params)}")
            unknown = set(r["stoich"]) - set(species)
            if unknown:
                raise ValidationError(f"{system.name}: reaction '{r['id']}' refers to unknown species {sorted(unknown)}")
            for key, name in r["params"].items():
                values[name] = system.param(name, params[key])
                if values[name] < 0 or (key in _MUST_BE_POSITIVE and values[name] <= 0):
                    raise ValidationError(f"{system.name}: parameter '{name}' has an invalid value {values[name]:g}")
            for name in r["refs"].values():
                if name not in species:
                    values[name] = system.param(name, "M")
                    if values[name] < 0:
                        raise ValidationError(f"{system.name}: concentration '{name}' must be non-negative")

        net = _Network(structure, values, backend)
        c0 = np.asarray(system.state["c"].value, dtype=float)
        if c0.shape != (len(species),) or np.any(c0 < 0):
            raise ValidationError(f"{system.name}: state 'c' must be one non-negative concentration per species")
        ones = np.ones(len(reactions))
        j_ref = ids.index(structure["flux_reaction"])

        c_star = net.steady_state(c0, ones)
        if c_star.max() > MAX_CONCENTRATION:
            worst = species[int(np.argmax(c_star))]
            raise ValidationError(
                f"{system.name}: the steady-state concentration of {worst} is {c_star.max():.3g} M, above any physical "
                f"limit ({MAX_CONCENTRATION:g} M): the model is far outside the dilute-solution regime it describes "
                f"(a rate constant, enzyme concentration or equilibrium constant is probably wrong by orders of magnitude)"
            )
        v_star = net.rates(c_star, ones)
        eigenvalues = np.linalg.eigvals(net.jacobian(c_star, ones)).real
        if eigenvalues.max() >= 0:
            raise ValidationError(f"{system.name}: the steady state is not stable (largest eigenvalue {eigenvalues.max():.3g} 1/s)")
        settling = 1.0 / np.abs(eigenvalues).min()

        src = self.name
        obs = {}
        for s, value in zip(species, c_star):
            obs[f"steady_{s}"] = Quantity(value, "M", source=src)
        for rid, value in zip(ids, v_star):
            obs[f"flux_{rid}"] = Quantity(value, "M/s", source=src)
        obs["pathway_flux"] = Quantity(v_star[j_ref], "M/s", source=src)
        obs["max_eigenvalue"] = Quantity(eigenvalues.max(), "1/s", source=src)
        obs["settling_time"] = Quantity(settling, "s", source=src)
        if abs(v_star[j_ref]) > 1e-9 * net.v_ref:                # at equilibrium J = 0 and ln|J| (hence control) is undefined
            for rid, value in zip(ids, _flux_control(net, c_star, j_ref)):
                obs[f"flux_control_{rid}"] = Quantity(value, "1", source=src)

        n_time = int(system.param("n_time", default=200))
        t = np.linspace(0.0, 10.0 * settling, n_time)
        obs["time"] = Quantity(t, "s", source=src)
        try:
            course = backend.integrate_ode(lambda _t, cc: net.rhs(cc, ones), c0, t)
        except RuntimeError as error:
            raise ValidationError(f"{system.name}: the time course could not be integrated ({error})") from None
        obs["concentrations"] = Quantity(course, "M", source=src)
        return system.evolve(
            observables=obs,
            state={"c": Quantity(c_star, "M", source=src)},
            dynamics="dc/dt = N v(c); steady state N v = 0; C_i = d ln J / d ln a_i",
        )


def _flux_control(net: _Network, c_star: np.ndarray, j_ref: int, h: float = 1e-4) -> list[float]:
    """Flux-control coefficients d ln|J| / d ln a_i by central difference on each reaction's activity, re-solving the
    steady state. |J| so that a pathway running in reverse (J < 0) has well-defined control too; the summation theorem
    holds either way."""
    out = []
    for i in range(len(net.reactions)):
        flux = []
        for factor in (1.0 + h, 1.0 - h):
            activity = np.ones(len(net.reactions))
            activity[i] = factor
            flux.append(net.rates(net.steady_state(c_star, activity), activity)[j_ref])
        c = (np.log(abs(flux[0])) - np.log(abs(flux[1]))) / (np.log(1.0 + h) - np.log(1.0 - h))
        out.append(0.0 if abs(c) < 1e-8 else float(c))
    return out
