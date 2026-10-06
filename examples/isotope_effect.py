"""Kinetic isotope effect for proton vs deuteron transfer, propagated quantum -> reaction.

Run from the project root:  python examples/isotope_effect.py
The toy double well is not fitted to a real molecule; read the *trend* (tunnelling plateau,
KIE growing on cooling), not the absolute numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry

PROTON, DEUTERON = 1.007276, 2.013553


def double_well(mass: float, temperature: float) -> ScientificSystem:
    return ScientificSystem(
        name="proton transfer",
        scale=Scale.QUANTUM,
        kind="quantum.double_well_1d",
        parameters={
            "mass": Quantity(mass, "amu"),
            "barrier_height": Quantity(0.50, "eV"),
            "half_separation": Quantity(0.40, "angstrom"),
            "reaction_energy": Quantity(0.05, "eV"),
            "temperature": Quantity(temperature, "K"),
        },
    )


pipeline = Pipeline(default_registry())
print(f"{'T (K)':>6} {'k_H (1/s)':>12} {'k_D (1/s)':>12} {'KIE':>8}   dominant level (H)")
for t in (150, 200, 250, 300, 350, 400):
    h = pipeline.run(double_well(PROTON, t), ["reaction"])
    d = pipeline.run(double_well(DEUTERON, t), ["reaction"])
    k_h, k_d = h.final.param("k_f"), d.final.param("k_f")
    level = h.trace[1].structure["derivation"]["dominant_level"]
    print(f"{t:>6} {k_h:>12.3e} {k_d:>12.3e} {k_h / k_d:>8.1f}   n={level}")
