"""Following an isotope effect from a proton hopping between two oxygens to the flux of a metabolic pathway.

The chain is  electronic -> molecular -> reaction -> biophysical -> biological.  The molecular scale computes the
intrinsic H/D isotope effect of the chemical step (~10, from zero-point energy alone). Whether that survives to the
pathway flux depends on THREE separate conditions, and breaking any one of them removes it:

  1. chemistry limits within the enzyme   (release must be faster than chemistry, or k_cat reports release)
  2. the enzyme is saturated              (substrate >> K_M, or the flux follows k_cat/K_M, which for a sticky substrate
                                           is just the binding rate)
  3. the enzyme controls the pathway      (flux-control coefficient near 1, or the flux is set elsewhere)

Table 1 breaks each condition in turn. Table 2 slows the transport step, which moves control away from the enzyme:
a first-order prediction from the flux-control coefficient, KIE_flux ~ KIE_kcat ^ C_enzyme, is shown next to the truth.
Control coefficients are local derivatives, so that prediction is only exact for small changes. A 10x change in k_cat is
not small: it can change which step limits the pathway, and then the prediction fails badly. That is part of the lesson.

The pathway is  external substrate -> S -> (enzyme) -> P -> drain.  All parameters are illustrative, not fitted to an organism.

Run from the project root:  python examples/pathway_isotope_effect.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry

PROTON, DEUTERON = 1.007276, 2.013553
ROUTE = ["electronic_structure", "molecular", "reaction", "biophysical", "biological"]
UNITS = {"k_on": "1/(M s)", "k_off": "1/s", "k_release": "1/s", "k_on_product": "1/(M s)", "substrate": "M",
         "enzyme_total": "M", "external_substrate": "M", "transport_rate": "1/s", "drain_vmax": "M/s", "drain_km": "M"}
BASE = dict(k_on=1e10, k_off=1e3, k_on_product=1e6, substrate=1e-4, enzyme_total=1e-8, transport_rate=1e4,
            drain_vmax=10.0, drain_km=1e-3, external_substrate=1e-1, k_release=1e12)


def system(mass: float, **context) -> ScientificSystem:
    p = {
        "morse_depth": (4.6, "eV"), "morse_alpha": (2.2, "1/angstrom"), "morse_r_eq": (0.96, "angstrom"),
        "coupling": (0.6, "eV"), "coupling_decay": (3.0, "1/angstrom"), "reference_distance": (2.5, "angstrom"),
        "oo_depth": (0.4, "eV"), "oo_alpha": (2.5, "1/angstrom"), "oo_equilibrium": (2.7, "angstrom"),
        "diabatic_offset": (0.05, "eV"), "particle_mass": (mass, "amu"), "heavy_atom_mass": (15.9949, "amu"),
        "temperature": (300.0, "K"),
    }
    p.update({f"context.{k}": (v, UNITS[k]) for k, v in {**BASE, **context}.items()})
    return ScientificSystem("pathway", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state_2d",
                            parameters={k: Quantity(v, u) for k, (v, u) in p.items()})


pipeline = Pipeline(default_registry())


def both(**context):
    h = pipeline.run(system(PROTON, **context), ROUTE)
    d = pipeline.run(system(DEUTERON, **context), ROUTE)
    biophysical = lambda r: r.trace[6]                       # the solved enzyme cycle
    return {
        "intrinsic": h.trace[3].param("k_f") / d.trace[3].param("k_f"),
        "kcat": biophysical(h).obs("kcat") / biophysical(d).obs("kcat"),
        "saturation": h.final.obs("steady_S") / biophysical(h).obs("KM"),
        "c_enzyme": h.final.obs("flux_control_enzyme"),
        "flux": h.final.obs("pathway_flux") / d.final.obs("pathway_flux"),
    }


print("Table 1: break each condition in turn (intrinsic H/D effect of the chemical step is ~10)\n")
print(f"{'regime':<46}{'KIE kcat':>9}{'S*/KM':>10}{'C_enzyme':>10}{'KIE flux':>10}")
regimes = {
    "all three conditions hold": {},
    "1 broken: slow product release": dict(k_release=1e3),
    "2 broken: unsaturated (X0 << KM)": dict(external_substrate=1e-5),
    "3 broken: transport limits the pathway": dict(transport_rate=1e-4, enzyme_total=1e-5),
}
for name, ctx in regimes.items():
    r = both(**ctx)
    print(f"{name:<46}{r['kcat']:>9.2f}{r['saturation']:>10.2g}{r['c_enzyme']:>10.3f}{r['flux']:>10.2f}")

print("\nTable 2: slowing transport moves control away from the enzyme (the other two conditions hold)\n")
print(f"{'transport (1/s)':>16}{'C_enzyme':>10}{'KIE kcat':>10}{'KIE flux':>10}{'first-order prediction':>24}")
for k_t in (1e4, 10.0, 1.0, 0.3, 0.1, 0.01):
    r = both(transport_rate=k_t)
    predicted = r["kcat"] ** r["c_enzyme"]
    print(f"{k_t:>16.2g}{r['c_enzyme']:>10.3f}{r['kcat']:>10.2f}{r['flux']:>10.2f}{predicted:>24.2f}")

print("\nOnly the first row of Table 1 lets the molecular-scale isotope effect through to the pathway flux.")
print("In Table 2 the flux effect falls as the enzyme gives up control. The first-order estimate works where one step clearly")
print("controls the pathway for both isotopes (top and bottom), and fails in between: at 0.3 1/s it predicts ~1.0 but the true")
print("effect is ~3.8. Control coefficients are local, and deuteration makes the enzyme 10x slower, which hands control back")
print("to it: the perturbation changes which step is limiting, so a local estimate made at H cannot see it.")
