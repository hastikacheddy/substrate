"""Calibrating the model engines against real quantum chemistry.

The valence-bond model runs in milliseconds, but its default parameters were illustrative: against real Hartree-Fock energies
for the Zundel cation its rates were wrong by 4.5 to 5.8 orders of magnitude. Here its parameters are fitted to the real
surface, and the example then checks, honestly, how far the cheap model can stand in for the expensive calculation:

  1. the fit and its diagnostics: where it matches the real surface, which parameters the data cannot pin down, and how well it
     predicts a heavy-atom distance it never saw;
  2. the answers that matter downstream (rates and isotope effects across nine orders of magnitude) against the real chain;
  3. what the model costs, against what the real calculation costs;
  4. uncertainty: why the fitted parameters must be drawn JOINTLY (they are strongly correlated), and why even then the spread is
     smaller than the model's real error, because the spread only measures how far the parameters can move.

Needs PySCF for the reference energies (WSL on Windows; see scripts/setup_qc_env.sh); they come from the on-disk cache after a
first run. Run from the project root:  python examples/calibrate_against_real_chemistry.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry
from substrate.calibration import calibrate_evb_2d
from substrate.engines.qc_scan import QCScanEngine, build_geometry
from substrate.molecules import zundel_cation
from substrate.qc import PySCFProgram, QCJob

PROTON, DEUTERON = 1.007276, 2.013553
ROUTE = ["electronic_structure", "quantum", "reaction"]

if not PySCFProgram().available():
    sys.exit("PySCF is not available. On Windows, run scripts/setup_qc_env.sh in WSL (see the README).")


def zundel(scan_distance: float, mass: float = PROTON) -> ScientificSystem:
    p = {"scan_distance": (scan_distance, "angstrom"), "x_extent": (0.7, "angstrom"), "distance_min": (2.3, "angstrom"),
         "distance_max": (3.0, "angstrom"), "n_x": (21, "1"), "n_r": (11, "1"), "particle_mass": (mass, "amu"),
         "heavy_atom_mass": (15.9949, "amu"), "temperature": (300.0, "K")}
    return ScientificSystem("Zundel cation", Scale.ELECTRONIC_STRUCTURE, "electronic.qc_scan_2d",
                            parameters={k: Quantity(v, u) for k, (v, u) in p.items()},
                            structure={"molecule": zundel_cation(), "method": {"theory": "hf", "basis": "6-31g*", "program": "pyscf"},
                                       "scan": {"mirror_symmetric": True}})


def default_model(scan_distance: float) -> ScientificSystem:
    """The model engine with its illustrative, uncalibrated default parameters."""
    p = {"morse_depth": (4.6, "eV"), "morse_alpha": (2.2, "1/angstrom"), "morse_r_eq": (0.96, "angstrom"), "coupling": (0.6, "eV"),
         "coupling_decay": (3.0, "1/angstrom"), "reference_distance": (2.5, "angstrom"), "oo_depth": (0.4, "eV"),
         "oo_alpha": (2.5, "1/angstrom"), "oo_equilibrium": (2.7, "angstrom"), "diabatic_offset": (0.0, "eV"),
         "scan_distance": (scan_distance, "angstrom"), "x_extent": (0.7, "angstrom"), "distance_min": (2.3, "angstrom"),
         "distance_max": (3.0, "angstrom"), "n_x": (41, "1"), "n_r": (21, "1"), "particle_mass": (PROTON, "amu"),
         "heavy_atom_mass": (15.9949, "amu"), "temperature": (300.0, "K")}
    return ScientificSystem("default model", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state_2d",
                            parameters={k: Quantity(v, u) for k, (v, u) in p.items()})


pipeline = Pipeline(default_registry())

print("1. Fit the model to the real surface\n")
target = QCScanEngine().solve(zundel(2.8), None)
started = time.time()
calibration = calibrate_evb_2d(target)
print(calibration.summary())
print(f"\n   (fit and leave-one-out validation: {time.time() - started:.0f} s)")

print("\n2. Does the cheap model give the real chain's answers?  (quantum route, 300 K)\n")
print(f"{'R (A)':>6}{'k_H real':>11}{'calibrated':>12}{'ratio':>7}{'uncalibrated':>14}{'ratio':>10}   {'KIE real':>8}{'KIE cal.':>9}")
for distance in (2.6, 2.7, 2.8, 2.9, 3.0):
    real = {m: pipeline.run(zundel(distance, m), ROUTE).final.param("k_f") for m in (PROTON, DEUTERON)}
    cal = {m: pipeline.run(calibration.system(scan_distance=distance, with_uncertainty=False,
                                              context={"particle_mass": (m, "amu")}), ROUTE).final.param("k_f")
           for m in (PROTON, DEUTERON)}
    raw = pipeline.run(default_model(distance), ROUTE).final.param("k_f")
    print(f"{distance:>6.1f}{real[PROTON]:>11.2e}{cal[PROTON]:>12.2e}{cal[PROTON] / real[PROTON]:>7.2f}{raw:>14.2e}"
          f"{raw / real[PROTON]:>10.1e}   {real[PROTON] / real[DEUTERON]:>8.0f}{cal[PROTON] / cal[DEUTERON]:>9.0f}")

print("\n3. What does each cost?")
started = time.time()
for _ in range(20):
    pipeline.run(calibration.system(scan_distance=2.8, with_uncertainty=False), ROUTE)
per_model_run = (time.time() - started) / 20
fresh = [QCJob(build_geometry(zundel_cation(), 0.1234 + 0.01 * i, 2.8 + 0.0123), 1, 0, "hf", "6-31g*") for i in range(24)]
started = time.time()
PySCFProgram().compute(fresh)
per_energy = (time.time() - started) / len(fresh)
print(f"   the calibrated model, whole chain to a rate: {per_model_run * 1e3:.0f} ms")
print(f"   one real energy (HF/6-31G*, this machine, a batch of {len(fresh)} fresh geometries including the WSL start-up): {per_energy:.2f} s")
print(f"   a real rate at a NEW heavy-atom distance needs {11} half-slice energies: ~{11 * per_energy:.0f} s, "
      f"so a 100-draw ensemble over that distance would take ~{100 * 11 * per_energy / 60:.0f} min against ~{100 * per_model_run:.0f} s calibrated")

print("\n4. Uncertainty at R = 2.8 A (60 draws): why the draws must be joint, and what the spread does not cover\n")
joint = pipeline.run(calibration.system(scan_distance=2.8), ROUTE, n_samples=60, seed=1)
independent_system = calibration.system(scan_distance=2.8)
independent_system.structure = {}                                          # the same sigmas, with the correlations thrown away
independent = pipeline.run(independent_system, ROUTE, n_samples=60, seed=1)
relative = lambda run: run.final.parameters["k_f"].sigma / run.final.parameters["k_f"].value
print(f"   draws from the joint covariance:      k_f spread {relative(joint):5.1%}   ({joint.ensemble['n_ok']}/60 valid)")
print(f"   independent draws (correlations lost): k_f spread {relative(independent):5.1%}   ({independent.ensemble['n_ok']}/60 valid)")
real_k = pipeline.run(zundel(2.8), ROUTE).final.param("k_f")
model_k = pipeline.run(calibration.system(scan_distance=2.8, with_uncertainty=False), ROUTE).final.param("k_f")
print(f"   the calibrated model's actual deviation from the real chain: {abs(model_k / real_k - 1):.0%}")
print("\nThe fit's parameter uncertainty is smaller than the model's real error. It measures how far the parameters can move before")
print("the fit visibly degrades, not what the model's fixed functional form cannot represent. Compare predictions directly.")
print("The fitted values are EFFECTIVE parameters (a coupling of several eV is not a physical valence-bond coupling) and apply to")
print("this molecule, method and distance range only.")
