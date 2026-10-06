"""Quantum -> reaction: rate constants from level structure and barrier transmission.

Sequential (incoherent) tunnelling with thermal level populations:

    k_f = (1 / Q_L) * sum_n  nu_n * P(E_n) * exp(-(E_n - E_0L) / kT)

    nu_n   attempt frequency of left-well level n (level-spacing derivative / h)
    P(E_n) barrier transmission probability at that level's energy
    Q_L    left-well partition function over the same levels

The reverse rate is fixed by detailed balance, k_r = k_f * Q_L / Q_R, so the network's
equilibrium constant is the exact ratio of well partition functions.
"""
from __future__ import annotations

import numpy as np

from ..base import Translator, ValidationIssue
from ..errors import ValidationError
from ..ir import Scale, ScientificSystem
from .reaction_network import two_state_network
from ..units import KB_EV

NAME = "quantum_to_reaction.sequential_tunneling"


class QuantumToReaction(Translator):
    name = NAME
    source = Scale.QUANTUM
    target = Scale.REACTION
    target_kind = "reaction.network"
    approximations = (
        "reaction coordinate is the single 1D coordinate: no friction, no recrossing, no coupling to other modes",
        "sequential tunnelling: wells are coupled weakly enough that a bath localises the particle between "
        "events (decoherence fast compared with the tunnel splitting)",
        "product side is a continuum for energy exchange (bath-assisted), so the rate is nu_n * P(E_n) per level",
        "thermal population only in well-localised levels below the barrier top; above-barrier flux is neglected",
        "reverse rate from detailed balance with well partition functions built from the same levels",
        "localised levels carry a systematic error bounded by the tunnel splitting (Dirichlet truncation)",
    )

    #: warn when the barrier above the reactant ground state is less than this many kT
    MIN_BARRIER_KT = 5.0

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        left = system.obs("levels_left", default=None)
        right = system.obs("levels_right", default=None)
        if left is None or right is None:
            return [ValidationIssue("error", "system has not been solved (levels_left/levels_right missing)")]
        temperature = system.param("temperature", "K", default=None)
        if temperature is None or temperature <= 0:
            return [ValidationIssue("error", "a positive 'temperature' parameter (K) is required")]
        if len(left) == 0 or len(right) == 0:
            issues.append(ValidationIssue(
                "error",
                "no localised level in one of the wells: the barrier is below the zero-point energy, "
                "so there is no reactant state and a rate constant is not defined",
            ))
            return issues
        kt = KB_EV * temperature
        gap = (system.obs("barrier_top") - left[0]) / kt
        if gap < self.MIN_BARRIER_KT:
            issues.append(ValidationIssue(
                "warning",
                f"barrier above the reactant ground state is only {gap:.1f} kT: above-barrier "
                f"thermal flux (neglected here) may matter",
            ))
        return issues

    def translate(self, system: ScientificSystem) -> ScientificSystem:
        temperature_q = system.parameters["temperature"]
        kt = KB_EV * temperature_q.value
        e_left = system.obs("levels_left")
        e_right = system.obs("levels_right")
        p = system.obs("transmission_left")
        nu = system.obs("attempt_frequency_left")

        ref = e_left[0]
        w_left = np.exp(-(e_left - ref) / kt)
        flux = nu * p * w_left
        q_left = w_left.sum()
        q_right = np.exp(-(e_right - ref) / kt).sum()
        k_f = flux.sum() / q_left
        k_eq = q_right / q_left
        k_r = k_f / k_eq
        if not (k_f > 0 and np.isfinite(k_r)):
            raise ValidationError(f"{system.name}: transmission is zero at every populated level, no rate constant")

        return two_state_network(
            system, k_f, k_r, temperature_q, self.name,
            {
                "flux_fraction_by_level": [float(f) for f in flux / flux.sum()] if flux.sum() > 0 else [],
                "dominant_level": int(np.argmax(flux)),
            },
        )
