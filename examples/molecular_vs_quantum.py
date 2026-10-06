"""Two routes from one electronic surface to a rate, and why their ratio is NOT a tunnelling factor.

  molecular route  heavy atoms free to relax (the relaxed barrier), classical transition-state theory, NO tunnelling
  quantum route    heavy-atom distance frozen at one value, proton tunnelling included

They make different approximations about different degrees of freedom. The quantum route's answer depends strongly on
where the O...O distance is frozen, and the molecular route cannot see tunnelling at all. The table shows the quantum
route frozen at three distances taken from the molecular route's own geometries: the transition state, the wells, and
a value in between. The spread between them is the size of the approximation, not a physical effect.

The isotope effect is a cleaner quantity. The molecular route's H/D ratio comes from zero-point energy alone (~10, the
textbook semiclassical value) and is a baseline. The quantum route gives anything from ~15 to thousands depending on
where the O...O distance is frozen: tunnelling is mass-sensitive, but how much it matters depends on how thick the
barrier is, and that is exactly what the frozen distance sets.

A proper answer needs the nuclear quantum problem in both coordinates, which this repository does not (yet) have.

Run from the project root:  python examples/molecular_vs_quantum.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry

PROTON, DEUTERON = 1.007276, 2.013553
MOLECULAR = ["electronic_structure", "molecular", "reaction"]
QUANTUM = ["electronic_structure", "quantum", "reaction"]


def oho(mass: float, scan_distance: float | None = None) -> ScientificSystem:
    p = {
        "morse_depth": (4.6, "eV"), "morse_alpha": (2.2, "1/angstrom"), "morse_r_eq": (0.96, "angstrom"),
        "coupling": (0.6, "eV"), "coupling_decay": (3.0, "1/angstrom"), "reference_distance": (2.5, "angstrom"),
        "oo_depth": (0.4, "eV"), "oo_alpha": (2.5, "1/angstrom"), "oo_equilibrium": (2.7, "angstrom"),
        "diabatic_offset": (0.05, "eV"), "particle_mass": (mass, "amu"), "heavy_atom_mass": (15.9949, "amu"),
        "temperature": (300.0, "K"),
    }
    if scan_distance is not None:
        p["scan_distance"] = (scan_distance, "angstrom")
    return ScientificSystem("O-H...O", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state_2d",
                            parameters={k: Quantity(v, u) for k, (v, u) in p.items()})


pipeline = Pipeline(default_registry())
rates = {}

tst = {m: pipeline.run(oho(m), MOLECULAR) for m in (PROTON, DEUTERON)}
mol = tst[PROTON].trace[2]
r_ts, r_well = mol.obs("geometry_ts")[1], mol.obs("geometry_reactant")[1]
rates["molecular route (relaxed, no tunnelling)"] = {m: tst[m].final.param("k_f") for m in tst}

for label, r in ((f"quantum route, O...O frozen at the TS distance ({r_ts:.2f} A)", r_ts),
                 ("quantum route, O...O frozen at 2.50 A", 2.5),
                 (f"quantum route, O...O frozen at the well distance ({r_well:.2f} A)", r_well)):
    rates[label] = {m: pipeline.run(oho(m, r), QUANTUM).final.param("k_f") for m in (PROTON, DEUTERON)}

print(f"relaxed O...O: {r_well:.3f} A in the wells -> {r_ts:.3f} A at the transition state (T = 300 K)\n")
print(f"{'route':<64}{'k_H (1/s)':>12}{'k_D (1/s)':>12}{'KIE':>8}")
for label, k in rates.items():
    print(f"{label:<64}{k[PROTON]:>12.2e}{k[DEUTERON]:>12.2e}{k[PROTON] / k[DEUTERON]:>8.1f}")
print("\nThe rates differ by orders of magnitude between routes: do not divide them to get a tunnelling factor.")
print("The molecular KIE (~10) is the zero-point-only baseline. The quantum KIE depends on the frozen distance: near the")
print("transition-state distance the barrier is nearly gone and tunnelling adds little; at the well distance it dominates.")
