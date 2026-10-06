"""Biophysical -> biological: an enzyme with computed kinetics becomes one reaction in a metabolic pathway.

The enzyme cycle's Michaelis-Menten parameters become the enzyme's rate law in a small pathway,

    external substrate  --transport-->  S  --enzyme-->  P  --drain-->  (out)

with total enzyme concentration E_T:  V_f = k_cat E_T,  K_s = K_M. If the enzyme is reversible (product rebinding in the
cycle) the rate law is the full thermodynamically consistent
    v = (V_f S/K_s - V_r P/K_p) / (1 + S/K_s + P/K_p),     V_r = k_cat,reverse E_T,  K_p = K_M,product,
which is exactly the steady-state turnover of the three-state cycle, so no information is lost in the translation.
Otherwise it is irreversible Michaelis-Menten.

Everything outside the enzyme is a biological input, passed as `context.*` parameters on the original system:

    context.enzyme_total (M)        total enzyme concentration in the compartment
    context.external_substrate (M)  the fixed external pool
    context.transport_rate (1/s)    exchange rate between the external pool and S: v = k (X0 - S)
    context.drain_vmax (M/s), context.drain_km (M)   downstream consumption of P, Michaelis-Menten
"""
from __future__ import annotations

from ..base import Translator, ValidationIssue
from ..engines.biological import KIND as PATHWAY_KIND
from ..engines.biological import pathway_structure
from ..engines.biophysical import KIND as ENZYME_KIND
from ..ir import Quantity, Scale, ScientificSystem

NAME = "biophysical_to_pathway.enzyme_in_pathway"

REQUIRED = {
    "context.enzyme_total": "M", "context.external_substrate": "M", "context.transport_rate": "1/s",
    "context.drain_vmax": "M/s", "context.drain_km": "M",
}

#: the quasi-steady-state enzyme rate law needs E_T well below K_M (+ S); warn when E_T exceeds this fraction of K_M
QSSA_FRACTION = 0.1


class BiophysicalToPathway(Translator):
    name = NAME
    source = Scale.BIOPHYSICAL
    source_kinds = (ENZYME_KIND,)
    target = Scale.BIOLOGICAL
    target_kind = PATHWAY_KIND
    approximations = (
        "the enzyme's kinetics are those of the isolated cycle from the lower scales, at the temperature it was computed for; "
        "the cellular environment (crowding, pH, regulators) is not represented",
        "quasi-steady-state enzyme kinetics: the enzyme's own intermediates equilibrate much faster than the metabolites change",
        "the pathway topology, transport, downstream consumption and enzyme abundance are supplied inputs",
        "a single enzyme species at a fixed total concentration: no expression, degradation or regulation",
    )

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        if system.obs("kcat", default=None) is None or system.obs("KM", default=None) is None:
            return [ValidationIssue("error", "system has not been solved (kcat/KM missing)")]
        problems = []
        for name, unit in REQUIRED.items():
            value = system.param(name, unit, default=None)
            if value is None:
                problems.append(f"missing context parameter '{name}' ({unit})")
            elif value <= 0:
                problems.append(f"'{name}' must be positive")
        if problems:
            return [ValidationIssue("error", "; ".join(problems))]
        issues = []
        enzyme, km = system.param("context.enzyme_total", "M"), system.obs("KM")
        if enzyme > QSSA_FRACTION * km:
            issues.append(ValidationIssue(
                "warning",
                f"total enzyme ({enzyme:.2g} M) exceeds {QSSA_FRACTION:g} x K_M ({km:.2g} M): the quasi-steady-state "
                f"enzyme rate law is questionable",
            ))
        return issues

    def translate(self, system: ScientificSystem) -> ScientificSystem:
        src = self.name
        e_total = system.param("context.enzyme_total", "M")
        reversible = system.obs("kcat_reverse", default=None) is not None
        params = {
            "enzyme_vf": Quantity(system.obs("kcat") * e_total, "M/s", source=src),
            "enzyme_ks": Quantity(system.obs("KM"), "M", source=src),
        }
        if reversible:
            params["enzyme_vr"] = Quantity(system.obs("kcat_reverse") * e_total, "M/s", source=src)
            params["enzyme_kp"] = Quantity(system.obs("KM_product"), "M", source=src)
        temperature = system.parameters.get("temperature")
        if temperature is not None:
            params["temperature"] = Quantity(temperature.value, "K", temperature.sigma, temperature.source)
        return ScientificSystem(
            name=f"{system.name} [pathway]",
            scale=Scale.BIOLOGICAL,
            kind=PATHWAY_KIND,
            parameters=params,
            state={"c": Quantity([0.0, 0.0], "M", source=src)},                # an empty compartment
            structure={**pathway_structure(reversible), "derivation": {"from": system.name, "reversible": reversible}},
            dynamics="dc/dt = N v(c)",
        )
