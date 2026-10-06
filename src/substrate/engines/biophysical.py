"""Biophysical scale: an enzyme turning over substrate, built from steps whose rates come from the scales below.

The model is a cyclic mechanism of enzyme states connected by first-order or pseudo-first-order steps (a binding step
has its rate multiplied by a ligand concentration). The standard instance is

    E + S  <->  ES  <->  EP  <->  E + P         (binding, chemistry, release)

where the chemical step ES <-> EP is exactly what the lower scales compute (a proton transfer, say) and the
binding and release rates are biophysical inputs, passed as `context.*` parameters.

For a given ligand concentration the steady state of the cycle is the null vector of its rate matrix. From it:
turnover per enzyme, population of each state, and, from the exact hyperbolic dependence on [S], k_cat and K_M.
Also reported: the binding free energy, and flux-control coefficients (which step limits k_cat).

This engine is small dense linear algebra and does not call the solver backend.
"""
from __future__ import annotations

import numpy as np

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem
from ..units import KB_EV

KIND = "biophysical.enzyme_cycle"
_PER_M_S = "1/(M s)"
_PER_S = "1/s"


def enzyme_cycle_structure(rebinding: bool = False) -> dict:
    """The structure of E + S <-> ES <-> EP <-> E + P. Rates named `context.*` are experiment-level inputs;
    `k_f` and `k_r` are the chemical step, supplied by the lower scales."""
    steps = [
        {"id": "bind", "from": "E", "to": "ES", "rate_parameter": "context.k_on", "ligand": "substrate"},
        {"id": "unbind", "from": "ES", "to": "E", "rate_parameter": "context.k_off"},
        {"id": "chem_f", "from": "ES", "to": "EP", "rate_parameter": "k_f"},
        {"id": "chem_r", "from": "EP", "to": "ES", "rate_parameter": "k_r"},
        {"id": "release", "from": "EP", "to": "E", "rate_parameter": "context.k_release"},
    ]
    if rebinding:
        steps.append({"id": "rebind", "from": "E", "to": "EP", "rate_parameter": "context.k_on_product", "ligand": "product"})
    return {
        "states": ["E", "ES", "EP"],
        "steps": steps,
        # Turnover is the net flux of product leaving the enzyme. Measured here, not across the chemical step, because
        # the chemistry can be orders of magnitude faster than everything else and near equilibrium: its net flux would be a
        # tiny difference of two huge numbers.
        "turnover": {"forward": "release", **({"reverse": "rebind"} if rebinding else {})},
        "binding": {"on": "bind", "off": "unbind"},
    }


def stationary_distribution(rates: np.ndarray) -> np.ndarray:
    """Stationary distribution of a continuous-time Markov chain, given rates[i, j] for the jump i -> j.

    Grassmann-Taksar-Heyman state reduction. It uses only additions, multiplications and divisions of non-negative
    numbers, so unlike solving Q pi = 0 it stays accurate when the rates span many orders of magnitude (a proton
    transfer at 1e13 1/s next to a release at 1e3 1/s). Raises ValueError if a state cannot reach the lower-numbered ones
    (reducible chain: two closed classes or an unreachable state).
    """
    a = np.array(rates, dtype=float)
    n = len(a)
    np.fill_diagonal(a, 0.0)
    out = np.zeros(n)
    for k in range(n - 1, 0, -1):
        out[k] = a[k, :k].sum()                       # total rate from k into the states that remain
        if not out[k] > 0.0:
            raise ValueError("reducible chain")
        a[:k, k] /= out[k]
        a[:k, :k] += np.outer(a[:k, k], a[k, :k])
        np.fill_diagonal(a[:k, :k], 0.0)
    pi = np.zeros(n)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ a[:k, k]
    return pi / pi.sum()


class _Cycle:
    """A cyclic mechanism with numeric rate constants: steady state, turnover, Michaelis-Menten parameters."""

    def __init__(self, structure: dict, rates: dict[str, float]):
        self.states = structure["states"]
        self.steps = structure["steps"]
        self.by_id = {s["id"]: s for s in self.steps}
        self.index = {s: i for i, s in enumerate(self.states)}
        self.turnover_pair = structure["turnover"]
        self.rates = dict(rates)

    def step_rate(self, step: dict, conc: dict[str, float]) -> float:
        k = self.rates[step["rate_parameter"]]
        return k * conc[step["ligand"]] if step.get("ligand") else k

    def populations(self, conc: dict[str, float]) -> np.ndarray:
        n = len(self.states)
        a = np.zeros((n, n))                             # a[i, j]: rate of the jump i -> j
        for step in self.steps:
            a[self.index[step["from"]], self.index[step["to"]]] += self.step_rate(step, conc)
        try:
            return stationary_distribution(a)
        except ValueError:
            raise ValidationError("the mechanism has a disconnected or absorbing state (a rate is zero)") from None

    def turnover(self, conc: dict[str, float]) -> float:
        """Net enzyme turnover (1/s): the flux of product leaving, minus any product coming back."""
        pi = self.populations(conc)
        fwd = self.by_id[self.turnover_pair["forward"]]
        flux = self.step_rate(fwd, conc) * pi[self.index[fwd["from"]]]
        if self.turnover_pair.get("reverse"):
            rev = self.by_id[self.turnover_pair["reverse"]]
            flux -= self.step_rate(rev, conc) * pi[self.index[rev["from"]]]
        return flux

    def michaelis_menten(self, ligand: str = "substrate") -> tuple[float, float]:
        """(k_cat, K_M) for turnover driven by one ligand with the other absent: the forward reaction for the substrate,
        the reverse reaction for the product. 1/v is exactly linear in 1/[ligand] when the ligand enters one step, so two
        points determine the hyperbola and a third checks it; a mechanism that fails the check is not Michaelis-Menten."""
        other = "product" if ligand == "substrate" else "substrate"
        sign = 1.0 if ligand == "substrate" else -1.0               # the reverse reaction has negative net turnover
        sub = [s for s in self.steps if s.get("ligand") == ligand]
        if not sub:
            raise ValidationError(f"no step involves the {ligand}")
        on = sum(self.rates[s["rate_parameter"]] for s in sub)
        s_ref = sum(self.step_rate(s, {"substrate": 1.0, "product": 0.0}) for s in self.steps if not s.get("ligand")) / on

        def v(s):
            return sign * self.turnover({ligand: s, other: 0.0})

        s1, s2, s3 = s_ref / 10.0, s_ref * 10.0, s_ref
        v1, v2 = v(s1), v(s2)
        if not (v1 > 0 and v2 > 0):
            raise ValidationError(f"the cycle has no net {'forward' if sign > 0 else 'reverse'} turnover: "
                                  f"a step needed to complete it has rate zero")
        slope = (1.0 / v1 - 1.0 / v2) / (1.0 / s1 - 1.0 / s2)           # K_M / k_cat
        intercept = 1.0 / v1 - slope / s1                               # 1 / k_cat
        not_mm = ValidationError(
            "turnover is not a Michaelis-Menten hyperbola in [S] (several substrate-dependent steps, e.g. substrate "
            "inhibition): k_cat and K_M are not defined for this mechanism"
        )
        if not (intercept > 0 and slope > 0):
            raise not_mm
        kcat, km = 1.0 / intercept, slope / intercept
        if abs(kcat * s3 / (km + s3) - v(s3)) > 1e-6 * v(s3):
            raise not_mm
        return kcat, km


class EnzymeCycleEngine(Engine):
    name = KIND
    scale = Scale.BIOPHYSICAL
    kinds = (KIND,)
    approximations = (
        "a single enzyme working at steady state: populations are the stationary vector of the cycle's rate matrix",
        "substrate and product concentrations are held fixed (no depletion), so only initial-rate kinetics is described",
        "every step is Markovian with a constant rate: no conformational heterogeneity, no cooperativity, no memory",
        "binding and release rates and concentrations are inputs; diffusion, membranes and the enzyme environment "
        "are not modelled, and the chemical step is the isolated model reaction from the lower scales",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        structure = system.structure
        rates = {}
        for step in structure["steps"]:
            name = step["rate_parameter"]
            if name in rates:
                continue
            rates[name] = system.param(name, _PER_M_S if step.get("ligand") else _PER_S)
        if any(k < 0 for k in rates.values()):
            raise ValidationError(f"{system.name}: negative rate constant")
        unknown = {s for st in structure["steps"] for s in (st["from"], st["to"])} - set(structure["states"])
        if unknown:
            raise ValidationError(f"{system.name}: steps refer to unknown state(s): {', '.join(sorted(unknown))}")
        substrate = system.param("context.substrate", "M")
        product = system.param("context.product", "M", default=0.0)
        if substrate < 0 or product < 0:
            raise ValidationError(f"{system.name}: concentrations must be non-negative")

        cycle = _Cycle(structure, rates)
        conc = {"substrate": substrate, "product": product}
        pi = cycle.populations(conc)
        kcat, km = cycle.michaelis_menten()

        src = self.name
        obs = {
            "turnover_rate": Quantity(cycle.turnover(conc), _PER_S, source=src),
            "kcat": Quantity(kcat, _PER_S, source=src),
            "KM": Quantity(km, "M", source=src),
            "kcat_over_KM": Quantity(kcat / km, _PER_M_S, source=src),
            "fraction_bound": Quantity(1.0 - pi[cycle.index["E"]], "1", source=src),
        }
        if "rebind" in cycle.by_id and rates[cycle.by_id["rebind"]["rate_parameter"]] > 0:
            # a reversible enzyme: the same cycle run backwards from product. With the forward pair these give the full
            # reversible rate law v = E_T (kcat_f S/KM_S - kcat_r P/KM_P) / (1 + S/KM_S + P/KM_P), exact for this mechanism.
            kcat_r, km_p = cycle.michaelis_menten("product")
            obs["kcat_reverse"] = Quantity(kcat_r, _PER_S, source=src)
            obs["KM_product"] = Quantity(km_p, "M", source=src)
        for state, p in zip(cycle.states, pi):
            obs[f"population_{state}"] = Quantity(p, "1", source=src)

        bind = structure["binding"]
        k_on, k_off = rates[cycle.by_id[bind["on"]]["rate_parameter"]], rates[cycle.by_id[bind["off"]]["rate_parameter"]]
        obs["K_d"] = Quantity(k_off / k_on, "M", source=src)
        temperature = system.param("temperature", "K", default=None)
        if temperature is not None:
            obs["delta_g_binding"] = Quantity(KB_EV * temperature * np.log(k_off / k_on), "eV", source=src)   # vs 1 M

        chem_f, chem_r = rates[cycle.by_id["chem_f"]["rate_parameter"]], rates[cycle.by_id["chem_r"]["rate_parameter"]]
        obs["chemical_equilibrium_constant"] = Quantity(chem_f / chem_r, "1", source=src)
        if "rebind" in cycle.by_id:
            overall = (k_on * chem_f * rates[cycle.by_id["release"]["rate_parameter"]]) / (
                k_off * chem_r * rates[cycle.by_id["rebind"]["rate_parameter"]])
            obs["overall_equilibrium_constant"] = Quantity(overall, "1", source=src)   # Haldane relation

        for name in rates:                                               # flux control of k_cat by each rate constant
            obs[f"control_{name.removeprefix('context.')}"] = Quantity(
                _control(structure, rates, name), "1", source=src)
        return system.evolve(
            observables=obs,
            dynamics="d(pi)/dt = Q pi = 0 for the enzyme cycle; k_cat, K_M from the hyperbola v([S])",
        )


def _control(structure: dict, rates: dict[str, float], name: str, h: float = 1e-4) -> float:
    """Flux-control coefficient d ln k_cat / d ln k_i by central difference (0 for a zero rate)."""
    if rates[name] == 0.0:
        return 0.0
    up, down = dict(rates), dict(rates)
    up[name] *= 1.0 + h
    down[name] *= 1.0 - h
    c = (np.log(_Cycle(structure, up).michaelis_menten()[0]) - np.log(_Cycle(structure, down).michaelis_menten()[0])) / (
        np.log(1.0 + h) - np.log(1.0 - h))
    return 0.0 if abs(c) < 1e-8 else float(c)
