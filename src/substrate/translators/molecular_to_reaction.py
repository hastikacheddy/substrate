"""Molecular -> reaction: classical transition-state-theory (Eyring) rate constants.

    k_f = (kT/h) exp(-dG_f / kT),    k_r = (kT/h) exp(-dG_r / kT)

with the free energies of activation from harmonic partition functions at the stationary points. Detailed balance
holds by construction: k_f / k_r = exp(-(G_product - G_reactant) / kT).

This route has no tunnelling. The validator measures how much that matters (hbar*omega_ts / kT) and says so.
"""
from __future__ import annotations

from ..base import Translator, ValidationIssue
from ..engines.molecular import KIND as MOLECULAR_KIND
from ..ir import Scale, ScientificSystem
from ..units import KB_EV
from .reaction_network import two_state_network

NAME = "molecular_to_reaction.tst"

#: above this, tunnelling raises the rate by more than ~20% (Bell/Wigner estimates agree near here)
TUNNELLING_WARN = 2.0
#: a free-energy barrier below this many kT is too low for TST's quasi-equilibrium assumption
MIN_BARRIER_KT = 5.0


class MolecularToReaction(Translator):
    name = NAME
    source = Scale.MOLECULAR
    source_kinds = (MOLECULAR_KIND,)
    target = Scale.REACTION
    target_kind = "reaction.network"
    approximations = (
        "classical transition-state theory: a quasi-equilibrium between reactant and TS, every crossing reactive",
        "no tunnelling correction (transmission coefficient 1) and no recrossing",
        "harmonic partition functions; translations and rotations cancel between reactant and TS",
        "a single reactant and a single product well; reverse rate from the product's own free-energy barrier",
    )

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        needed = ("rate_tst_forward", "rate_tst_reverse", "delta_g_forward", "delta_g_reverse", "tunnelling_parameter")
        if any(system.obs(n, default=None) is None for n in needed):
            return [ValidationIssue("error", "system has not been solved (TST observables missing)")]
        temperature = system.param("temperature", "K", default=None)
        if temperature is None or temperature <= 0:
            return [ValidationIssue("error", "a positive 'temperature' parameter (K) is required")]
        dg_f, dg_r = system.obs("delta_g_forward"), system.obs("delta_g_reverse")
        if dg_f <= 0 or dg_r <= 0:
            return [ValidationIssue(
                "error",
                f"the free-energy barrier is not positive (forward {dg_f:.3f} eV, reverse {dg_r:.3f} eV): "
                f"with zero-point energy the transition state is no higher than a well, so TST gives no rate",
            )]
        issues = []
        kt = KB_EV * temperature
        if min(dg_f, dg_r) < MIN_BARRIER_KT * kt:
            issues.append(ValidationIssue(
                "warning",
                f"a free-energy barrier is under {MIN_BARRIER_KT:g} kT ({min(dg_f, dg_r) / kt:.1f} kT): TST's "
                f"quasi-equilibrium assumption is weak",
            ))
        u = system.obs("tunnelling_parameter")
        if u > TUNNELLING_WARN:
            issues.append(ValidationIssue(
                "warning",
                f"hbar*omega/kT at the transition state is {u:.1f} (> {TUNNELLING_WARN:g}): tunnelling is significant "
                f"and classical TST underestimates the rate; see the quantum route",
            ))
        return issues

    def translate(self, system: ScientificSystem) -> ScientificSystem:
        return two_state_network(
            system,
            system.obs("rate_tst_forward"),
            system.obs("rate_tst_reverse"),
            system.parameters["temperature"],
            self.name,
            {
                "route": "classical transition-state theory (no tunnelling)",
                "delta_g_forward_eV": float(system.obs("delta_g_forward")),
                "delta_g_reverse_eV": float(system.obs("delta_g_reverse")),
                "tunnelling_parameter": float(system.obs("tunnelling_parameter")),
            },
        )
