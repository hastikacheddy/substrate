"""Reaction -> biophysical: the two-state reaction becomes the chemical step of an enzyme cycle.

The reaction system's A <-> B (rate constants k_f, k_r from whichever lower-scale route produced them) is placed
between the enzyme-substrate complex (A = ES) and the enzyme-product complex (B = EP):

    E + S <-> ES  <->  EP <-> E + P            ES <-> EP is the chemistry computed below

The binding and release rates and the concentrations are not computed by any lower scale; they are experiment-level
inputs, supplied as `context.*` parameters on the original system and carried here by the pipeline:

    context.k_on (1/(M s))   context.k_off (1/s)   context.k_release (1/s)   context.substrate (M)
    optional: context.product (M, default 0), context.k_on_product (1/(M s), enables product rebinding)
"""
from __future__ import annotations

from ..base import Translator, ValidationIssue
from ..engines.biophysical import KIND as ENZYME_KIND
from ..engines.biophysical import enzyme_cycle_structure
from ..ir import Quantity, Scale, ScientificSystem
from ..units import H_EV_S, KB_EV

NAME = "reaction_to_enzyme.chemical_step"

REQUIRED = {
    "context.k_on": "1/(M s)", "context.k_off": "1/s", "context.k_release": "1/s", "context.substrate": "M",
}
OPTIONAL = {"context.product": "M", "context.k_on_product": "1/(M s)"}

#: association rates above this (1/(M s)) exceed what diffusion allows for a protein-ligand encounter
DIFFUSION_LIMIT = 1e10
#: a chemical rate above this fraction of the TST frequency limit kT/h means the step is essentially barrierless
BARRIERLESS_FRACTION = 0.1


class ReactionToEnzyme(Translator):
    name = NAME
    source = Scale.REACTION
    source_kinds = ("reaction.network",)
    target = Scale.BIOPHYSICAL
    target_kind = ENZYME_KIND
    approximations = (
        "the chemical step is the isolated model reaction: the enzyme's electrostatic environment, which changes real "
        "barriers, is not represented",
        "binding, release and concentrations are supplied inputs, not computed from any lower scale",
        "one substrate, one product and a single chemical step; every step is a Markov jump with a constant rate",
        "the chemistry is a rate constant: this is valid only if the chemical step is slow compared with vibrational "
        "relaxation within the complex",
    )

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        s = system.structure
        if s.get("species") != ["A", "B"] or len(s.get("reactions", [])) != 2:
            return [ValidationIssue("error", "only a two-state A <-> B reaction can serve as the chemical step")]
        for name in ("k_f", "k_r"):
            if (system.param(name, "1/s", default=None) or 0) <= 0:
                return [ValidationIssue("error", f"the reaction needs a positive '{name}' (1/s)")]
        problems = []
        for name, unit in REQUIRED.items():
            value = system.param(name, unit, default=None)
            if value is None:
                problems.append(f"missing context parameter '{name}' ({unit})")
            elif value <= 0:
                problems.append(f"'{name}' must be positive")
        for name, unit in OPTIONAL.items():
            value = system.param(name, unit, default=0.0)
            if value < 0:
                problems.append(f"'{name}' must not be negative")
        if problems:
            return [ValidationIssue("error", "; ".join(problems))]

        issues = []
        if system.param("context.k_on", "1/(M s)") > DIFFUSION_LIMIT or system.param("context.k_on_product", "1/(M s)", default=0.0) > DIFFUSION_LIMIT:
            issues.append(ValidationIssue(
                "warning", f"an association rate exceeds the diffusion limit (~{DIFFUSION_LIMIT:.0e} 1/(M s))"))
        temperature = system.param("temperature", "K", default=None)
        if temperature is not None:
            limit = KB_EV * temperature / H_EV_S
            fastest = max(system.param("k_f", "1/s"), system.param("k_r", "1/s"))
            if fastest > BARRIERLESS_FRACTION * limit:
                issues.append(ValidationIssue(
                    "warning",
                    f"the chemical rate ({fastest:.1e} 1/s) is within {1 / BARRIERLESS_FRACTION:.0f}x of the "
                    f"transition-state frequency limit kT/h ({limit:.1e} 1/s): the step is nearly barrierless and a "
                    f"constant-rate Markov description is doubtful",
                ))
        return issues

    def translate(self, system: ScientificSystem) -> ScientificSystem:
        src = self.name
        rebinding = system.param("context.k_on_product", "1/(M s)", default=0.0) > 0
        params = {
            "k_f": Quantity(system.param("k_f"), "1/s", system.parameters["k_f"].sigma, src),
            "k_r": Quantity(system.param("k_r"), "1/s", system.parameters["k_r"].sigma, src),
        }
        temperature = system.parameters.get("temperature")
        if temperature is not None:
            params["temperature"] = Quantity(temperature.value, "K", temperature.sigma, temperature.source)
        return ScientificSystem(
            name=f"{system.name} [enzyme]",
            scale=Scale.BIOPHYSICAL,
            kind=ENZYME_KIND,
            parameters=params,
            structure={**enzyme_cycle_structure(rebinding), "derivation": {"from": system.name}},
            dynamics="steady state of the enzyme cycle",
        )
