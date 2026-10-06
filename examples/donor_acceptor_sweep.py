"""How the heavy-atom geometry steers the kinetics: electronic structure -> quantum -> reaction.

Sweeps the O...O distance of a model O-H...O proton transfer. At each distance the potential-energy surface
is *computed* (two-state valence-bond Hamiltonian, diagonalised at every proton position), handed to the
nuclear quantum solver, and turned into rate constants. When the O...O distance is short enough that the
barrier falls below the zero-point energy, the pipeline refuses to produce a rate.

Run from the project root:  python examples/donor_acceptor_sweep.py
The model is not fitted to a real molecule; read the trends, not the absolute numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, Quantity, Scale, ScientificSystem, ValidationError, default_registry

PROTON, DEUTERON = 1.007276, 2.013553


def oho_complex(distance: float, mass: float) -> ScientificSystem:
    params = {
        "donor_acceptor_distance": (distance, "angstrom"),
        "morse_depth": (4.6, "eV"),
        "morse_alpha": (2.2, "1/angstrom"),
        "morse_r_eq": (0.96, "angstrom"),
        "coupling": (0.6, "eV"),
        "diabatic_offset": (0.05, "eV"),
        "particle_mass": (mass, "amu"),
        "temperature": (300.0, "K"),
    }
    return ScientificSystem(
        "O-H...O", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state",
        parameters={k: Quantity(v, u) for k, (v, u) in params.items()},
    )


pipeline = Pipeline(default_registry())
print(f"{'R (A)':>6} {'barrier (eV)':>13} {'ZPE (eV)':>9} {'k_H (1/s)':>11} {'k_D (1/s)':>11} {'KIE':>8}")
for r in (2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60, 2.70):
    try:
        h = pipeline.run(oho_complex(r, PROTON), ["reaction"])
        d = pipeline.run(oho_complex(r, DEUTERON), ["reaction"])
    except ValidationError as e:
        print(f"{r:>6.2f}  no rate: {str(e).split(': ', 1)[1][:78]}...")
        continue
    k_h, k_d = h.final.param("k_f"), d.final.param("k_f")
    print(
        f"{r:>6.2f} {h.trace[0].obs('classical_barrier'):>13.3f} {h.trace[2].obs('zpe_left'):>9.3f} "
        f"{k_h:>11.3e} {k_d:>11.3e} {k_h / k_d:>8.1f}"
    )
