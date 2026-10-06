"""The two-state reaction network (A <-> B) that every route into the reaction scale produces."""
from __future__ import annotations

from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem


def two_state_network(
    source: ScientificSystem, k_f: float, k_r: float, temperature: Quantity, producer: str, derivation: dict
) -> ScientificSystem:
    """Reaction system for A <-> B with the given rate constants (1/s), starting entirely in A.

    `derivation` is recorded verbatim in the system's structure so the route that produced the rates
    stays inspectable (which levels carried the flux, which free energies were used, ...).
    """
    if not (k_f > 0 and k_r > 0):
        raise ValidationError(f"{source.name}: rate constants must be positive (k_f={k_f:.3g}, k_r={k_r:.3g})")
    return ScientificSystem(
        name=f"{source.name} [reaction]",
        scale=Scale.REACTION,
        kind="reaction.network",
        parameters={
            "k_f": Quantity(k_f, "1/s", source=producer),
            "k_r": Quantity(k_r, "1/s", source=producer),
            "K_eq": Quantity(k_f / k_r, "1", source=producer),
            "temperature": Quantity(temperature.value, "K", temperature.sigma, temperature.source),
            "t_end": Quantity(10.0 / (k_f + k_r), "s", source=producer),
            "n_time": Quantity(200, "1", source=producer),
        },
        state={"c": Quantity([1.0, 0.0], "population", source=producer)},
        structure={
            "species": ["A", "B"],
            "reactions": [
                {"id": "A->B", "reactants": {"A": 1}, "products": {"B": 1}, "rate_parameter": "k_f"},
                {"id": "B->A", "reactants": {"B": 1}, "products": {"A": 1}, "rate_parameter": "k_r"},
            ],
            "derivation": {"from": source.name, **derivation},
        },
        dynamics="dc/dt = S . v(c)",
    )
