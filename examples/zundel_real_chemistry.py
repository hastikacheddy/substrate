"""A proton shared between two waters, with real quantum chemistry instead of a model.

The Zundel cation H5O2+ is computed with Hartree-Fock / 6-31G* (PySCF) on a rigid-group grid of proton position x and
oxygen-oxygen distance R. The same pipeline that ran on the model Hamiltonians then runs on it unchanged.

What it shows:
  1. the barrier to moving the proton grows steeply with the O...O distance (real chemistry confirming the model's central
     assumption), and the uncalibrated model's numbers are far from the real ones: its parameters were illustrative;
  2. on the relaxed surface the oxygens approach and the proton is shared, so there is no proton-transfer reaction: the
     molecular route refuses, and says why;
  3. with the O...O distance held fixed (an enzyme active site does this), the quantum route gives rates that fall by a factor
     of ~4x10^8 between 2.6 and 3.0 angstrom, with an H/D isotope effect growing from ~9 to ~10^4 as tunnelling takes over.

Needs PySCF. On Windows that means the WSL environment from scripts/setup_qc_env.sh. Energies are cached on disk by content:
the first run takes a couple of minutes, later runs seconds.

Run from the project root:  python examples/zundel_real_chemistry.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, Quantity, Scale, ScientificSystem, ValidationError, default_registry
from substrate.engines.electronic import EVBProtonTransferEngine
from substrate.backends import ClassicalBackend
from substrate.molecules import zundel_cation
from substrate.qc import PySCFProgram

PROTON, DEUTERON = 1.007276, 2.013553

if not PySCFProgram().available():
    sys.exit("PySCF is not available. On Windows, run scripts/setup_qc_env.sh in WSL (see the README).")


def zundel(scan_distance: float, mass: float = PROTON) -> ScientificSystem:
    p = {
        "scan_distance": (scan_distance, "angstrom"), "x_extent": (0.7, "angstrom"),
        "distance_min": (2.3, "angstrom"), "distance_max": (3.0, "angstrom"), "n_x": (21, "1"), "n_r": (11, "1"),
        "particle_mass": (mass, "amu"), "heavy_atom_mass": (15.9949, "amu"), "temperature": (300.0, "K"),
    }
    return ScientificSystem(
        "Zundel cation", Scale.ELECTRONIC_STRUCTURE, "electronic.qc_scan_2d",
        parameters={k: Quantity(v, u) for k, (v, u) in p.items()},
        structure={"molecule": zundel_cation(), "method": {"theory": "hf", "basis": "6-31g*", "program": "pyscf"},
                   "scan": {"mirror_symmetric": True}},
    )


def model_barrier(distance: float) -> float:
    """The (uncalibrated) valence-bond model's rigid barrier at the same O...O distance."""
    p = {"donor_acceptor_distance": (distance, "angstrom"), "morse_depth": (4.6, "eV"), "morse_alpha": (2.2, "1/angstrom"),
         "morse_r_eq": (0.96, "angstrom"), "coupling": (0.6, "eV"), "diabatic_offset": (0.0, "eV")}
    system = ScientificSystem("model", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state",
                              parameters={k: Quantity(v, u) for k, (v, u) in p.items()})
    return EVBProtonTransferEngine().solve(system, ClassicalBackend()).obs("classical_barrier")


pipeline = Pipeline(default_registry())
started = time.time()
scan = pipeline.run(zundel(2.8), ["electronic_structure"]).final
print(f"Zundel cation, HF/6-31G*: {scan.obs('calculations_run'):.0f} energies computed, "
      f"{scan.obs('calculations_cached'):.0f} from the cache ({time.time() - started:.0f} s)\n")

print("1. Rigid barrier vs O...O distance (eV)")
print(f"{'R (A)':>8}{'real HF/6-31G*':>16}{'uncalibrated model':>20}")
for distance in (2.5, 2.6, 2.7, 2.8, 2.9, 3.0):
    real = pipeline.run(zundel(distance), ["electronic_structure"]).final
    print(f"{distance:>8.1f}{real.obs('classical_barrier', default=float('nan')):>16.3f}{model_barrier(distance):>20.3f}")

print("\n2. Let the oxygens relax (molecular route)")
try:
    pipeline.run(zundel(2.8), ["electronic_structure", "molecular", "reaction"])
except ValidationError as error:
    print("   refused:", str(error).split(": ", 1)[1])

print("\n3. Hold the O...O distance fixed (quantum route): rates and isotope effect at 300 K")
print(f"{'R (A)':>8}{'barrier (eV)':>14}{'k_H (1/s)':>13}{'k_D (1/s)':>13}{'KIE':>8}")
route = ["electronic_structure", "quantum", "reaction"]
for distance in (2.6, 2.7, 2.8, 2.9, 3.0):
    barrier = pipeline.run(zundel(distance), ["electronic_structure"]).final.obs("classical_barrier", default=float("nan"))
    try:
        k_h = pipeline.run(zundel(distance, PROTON), route).final.param("k_f")
        k_d = pipeline.run(zundel(distance, DEUTERON), route).final.param("k_f")
    except ValidationError as error:
        print(f"{distance:>8.1f}{barrier:>14.3f}  no rate: {str(error).split(': ', 1)[1][:75]}...")
        continue
    print(f"{distance:>8.1f}{barrier:>14.3f}{k_h:>13.2e}{k_d:>13.2e}{k_h / k_d:>8.0f}")

print("\nThe model engines' parameters were illustrative and uncalibrated; this is what the same chain gives on real energies.")
print("The scan is rigid (bond lengths and angles inside each water-like unit are frozen) and gas-phase, at Hartree-Fock level:")
print("treat the numbers as the output of that stated approximation, not as the Zundel cation's true rates.")
