"""Why an enzyme's measured isotope effect is usually smaller than the chemistry's: commitment masking.

A proton transfer sits inside an enzyme cycle  E + S <-> ES <-> EP -> E + P.  The molecular scale computes the *intrinsic*
isotope effect of the chemical step ES <-> EP (from zero-point energy alone, ~10 for O-H / O-D). What an experimentalist
measures is the effect on k_cat or k_cat/K_M, and that depends on whether chemistry is the slow step:

  - product release much faster than chemistry  ->  k_cat reports the chemistry: the full intrinsic effect shows
  - product release much slower than chemistry  ->  the chemistry equilibrates and is invisible: the effect vanishes

The flux-control coefficient of the chemical step says which regime you are in.

k_cat/K_M is a different case: it is set by what happens up to the first irreversible step, so it reports the chemistry only
if a bound substrate is likely to dissociate before it reacts (k_off comparable to or larger than the chemical rate).
Otherwise the substrate is "sticky" and k_cat/K_M is simply the binding rate, with no isotope effect at all. A second table
varies k_off to show this.

The chemistry here is the molecular route's (heavy atoms relax, classical TST). All parameters are illustrative, not
fitted to an enzyme.

Run from the project root:  python examples/enzyme_isotope_effects.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry

PROTON, DEUTERON = 1.007276, 2.013553
ROUTE = ["electronic_structure", "molecular", "reaction", "biophysical"]


def enzyme_system(mass: float, k_release: float, k_on: float = 1e7, k_off: float = 1e3) -> ScientificSystem:
    p = {
        "morse_depth": (4.6, "eV"), "morse_alpha": (2.2, "1/angstrom"), "morse_r_eq": (0.96, "angstrom"),
        "coupling": (0.6, "eV"), "coupling_decay": (3.0, "1/angstrom"), "reference_distance": (2.5, "angstrom"),
        "oo_depth": (0.4, "eV"), "oo_alpha": (2.5, "1/angstrom"), "oo_equilibrium": (2.7, "angstrom"),
        "diabatic_offset": (0.05, "eV"), "particle_mass": (mass, "amu"), "heavy_atom_mass": (15.9949, "amu"),
        "temperature": (300.0, "K"),
        # experiment-level biophysical context: carried across every scale by the pipeline
        "context.k_on": (k_on, "1/(M s)"), "context.k_off": (k_off, "1/s"),
        "context.k_release": (k_release, "1/s"), "context.substrate": (1e-4, "M"),
    }
    return ScientificSystem("enzyme", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state_2d",
                            parameters={k: Quantity(v, u) for k, (v, u) in p.items()})


def both_isotopes(pipeline, **kw):
    h = pipeline.run(enzyme_system(PROTON, **kw), ROUTE)
    d = pipeline.run(enzyme_system(DEUTERON, **kw), ROUTE)
    return h, d, h.trace[3].param("k_f") / d.trace[3].param("k_f")         # trace[3]: the reaction system


pipeline = Pipeline(default_registry())

print("Proton transfer inside an enzyme cycle (300 K)\n")
print("Table 1: vary product release (substrate binds tightly: k_off = 1e3 1/s, far below the chemical rate)")
print(f"{'k_release (1/s)':>16} {'intrinsic KIE':>14} {'KIE on kcat':>12} {'chemistry control of kcat':>26}")
for release in (1e12, 1e10, 1e8, 1e6, 1e4, 1e2):
    h, d, intrinsic = both_isotopes(pipeline, k_release=release)
    ho, do = h.final.observables, d.final.observables
    # Speeding up "the chemistry" means scaling k_f and k_r together (the equilibrium is unchanged), so its control is
    # the signed sum: 1 when chemistry limits k_cat, 0 once it has equilibrated and no longer matters.
    control = ho["control_k_f"].value + ho["control_k_r"].value
    print(f"{release:>16.0e} {intrinsic:>14.1f} {ho['kcat'].value / do['kcat'].value:>12.2f} {control:>26.2f}")

print("\nTable 2: vary substrate stickiness (release fast; k_on at the diffusion limit 1e10, K_d = k_off / k_on)")
print(f"{'k_off (1/s)':>16} {'K_d (M)':>10} {'intrinsic KIE':>14} {'KIE on kcat/KM':>15}")
for k_off in (1e4, 1e6, 1e7, 1e8, 1e9):
    h, d, intrinsic = both_isotopes(pipeline, k_release=1e12, k_on=1e10, k_off=k_off)
    print(f"{k_off:>16.0e} {h.final.obs('K_d'):>10.0e} {intrinsic:>14.1f} "
          f"{h.final.obs('kcat_over_KM') / d.final.obs('kcat_over_KM'):>15.2f}")

print("\nTable 1: a fast release reports the chemistry (KIE on kcat -> the intrinsic ~10); a slow release hides it (-> 1).")
print("Table 2: a sticky substrate (k_off << chemical rate) gives kcat/KM = k_on and no isotope effect; once the substrate")
print("can leave faster than it reacts, kcat/KM reports the chemistry.")
