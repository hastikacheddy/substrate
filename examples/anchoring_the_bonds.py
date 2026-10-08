"""Is the failure of the fitted parameters to transfer a consequence of letting them be effective? Anchor the bonds and see.

The calibration study fits twelve parameters per surface, several of them correlated and some on their bounds, then finds that they do not
carry from one molecule or method to another. One reading is that the model needs effective parameters, so there is nothing physical to carry.
Another is that a flexible fit simply wanders along its valleys and the physics would carry if it were pinned. This script separates them by
holding each X-H bond at the free diatomic's Morse curve (`substrate.diatomics`: spectroscopic constants, not fitted) and refitting only the
rest (coupling and its decay, the heavy-atom well, and the energy offset):

    shipped    the model as the reference files declare it (shared bonds, or separate for chloride-HF)
    shape      equilibrium length and width fixed at the free diatomic's, depth free
    full       equilibrium length, width and depth fixed

For each it reports how well the surfaces are still fitted and how the cross-prediction errors of the study's kinds of pair change. The same
24 references (the study's 21 and the three chloride-HF ones) are used.

    python examples/anchoring_the_bonds.py                  # reads the scans from the energy cache; about two minutes
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, default_registry, load_experiment
from substrate.calibration import FitSettings
from substrate.diatomics import DIATOMICS, anchored_bounds, morse_parameters, needs_separate_bonds
from substrate.transfer import baselines, cross_predict, fit_settings, make_reference

REFERENCES = Path(__file__).resolve().parent.parent / "experiments" / "references"
COMMON_REFERENCE_DISTANCE = 2.7
TIERS = ("shipped", "shape", "full")

pipeline = Pipeline(default_registry())
experiments = {}
for path in sorted(REFERENCES.glob("*.yaml")):
    experiment = load_experiment(path)
    experiments[path.stem] = (experiment, pipeline.run(experiment.system, experiment.propagation).final)
names = list(experiments)
print(f"{len(names)} references read from the cache\n")


def settings_for(tier: str, name: str) -> FitSettings:
    experiment, solved = experiments[name]
    base = fit_settings(experiment, FitSettings(reference_distance=COMMON_REFERENCE_DISTANCE))
    if tier == "shipped":
        return base
    molecule = solved.structure["molecule"]
    asymmetric_atoms = needs_separate_bonds(molecule)
    # an asymmetric surface with one element at both ends still needs the offset (always free there); the bond model only changes if the elements differ
    return replace(base, bonds="separate" if asymmetric_atoms else "shared", bounds=anchored_bounds(molecule, depth=(tier == "full")))


started = time.time()
references = {tier: {} for tier in TIERS}
for tier in TIERS:
    for name in names:
        experiment, solved = experiments[name]
        references[tier][name] = make_reference(name, experiment.system, solved, settings_for(tier, name))
print(f"{len(TIERS) * len(names)} fits in {time.time() - started:.0f} s\n")


def theory(reference) -> str:
    return reference.method.split("/")[0]


# -- 1. how far the fitted bonds are from the free diatomics ---------------------------------------------------------------------------------
print("1. The shipped fits' bond parameters against the free diatomics (donor bond; acceptor in brackets where it has its own)\n")
print(f"   {'reference':<26}{'r_eq fit':>9}{'free':>7}{'alpha fit':>10}{'free':>7}{'depth fit':>10}{'free':>7}")
deltas = {"morse_r_eq": [], "morse_alpha": [], "morse_depth": []}
for name in names:
    reference = references["shipped"][name]
    molecule = experiments[name][1].structure["molecule"]
    donor = molecule["atoms"][molecule["donor"]][0]
    free, fit = morse_parameters(donor), reference.calibration.parameters
    for key in deltas:
        deltas[key].append(fit[key] / free[key])
    print(f"   {name:<26}{fit['morse_r_eq']:>9.3f}{free['morse_r_eq']:>7.3f}{fit['morse_alpha']:>10.2f}{free['morse_alpha']:>7.2f}{fit['morse_depth']:>10.2f}{free['morse_depth']:>7.2f}"
          + ("" if "acceptor_morse_r_eq" not in fit else f"   [{fit['acceptor_morse_r_eq']:.3f} {fit['acceptor_morse_alpha']:.2f} {fit['acceptor_morse_depth']:.2f}]"))
print()
for key, values in deltas.items():
    print(f"   {key:<12} fitted / free: median {np.median(values):.2f}, range {min(values):.2f}-{max(values):.2f}")

# -- 2. what anchoring costs in fit quality -----------------------------------------------------------------------------------------------------
print("\n2. Own-fit rmse (eV, over each reference's declared window) and wrong well counts\n")
print(f"   {'reference':<26}{'window':>7}" + "".join(f"{t:>11}{'wells':>6}" for t in TIERS))
own = {tier: {} for tier in TIERS}
for name in names:
    row = f"   {name:<26}{references['shipped'][name].window_ev:>7g}"
    for tier in TIERS:
        reference = references[tier][name]
        prediction = cross_predict(reference.calibration, reference)
        own[tier][name] = (prediction.rmse_window, prediction.wells_wrong)
        row += f"{prediction.rmse_window:>11.4f}{prediction.wells_wrong:>6}"
    print(row)
print()
for tier in TIERS:
    values = [v[0] for v in own[tier].values()]
    wells = sum(v[1] for v in own[tier].values())
    print(f"   {tier:<8} median {np.median(values):.4f}, range {min(values):.4f}-{max(values):.4f}; wrong well counts over all {len(names)} references: {wells}")

# -- 3. transfer ---------------------------------------------------------------------------------------------------------------------------------------
print("\n3. Cross-prediction (source parameters unchanged, best constant shift) by kind of pair; 'x const' counts pairs worse than the target's constant guess\n")


def kind(source, target):
    s, t = references["shipped"][source], references["shipped"][target]
    if source == target:
        return "own"
    if s.molecule == t.molecule:
        if theory(s) == theory(t):
            return "same molecule, other basis"
        pair = {theory(s), theory(t)}
        return "same molecule, HF <-> B3LYP/MP2" if "hf" in pair else "same molecule, B3LYP <-> MP2"
    return "other molecule, same method" if theory(s) == theory(t) else "other molecule, other method"


KINDS = ["own", "same molecule, other basis", "same molecule, B3LYP <-> MP2", "same molecule, HF <-> B3LYP/MP2", "other molecule, same method", "other molecule, other method"]
print(f"   {'':<36}" + "".join(f"{t:^34}" for t in TIERS))
print(f"   {'kind of pair':<36}" + "".join(f"{'n':>4}{'median':>9}{'range':>16}{'x const':>5}" for _ in TIERS))
results = {tier: {k: [] for k in KINDS} for tier in TIERS}
for tier in TIERS:
    constants = {name: baselines(references[tier][name])["constant_rmse"] for name in names}
    for source in names:
        for target in names:
            prediction = cross_predict(references[tier][source].calibration, references[tier][target])
            results[tier][kind(source, target)].append((prediction.rmse_window, prediction.rmse_window / constants[target] > 1.0))
for k in KINDS:
    row = f"   {k:<36}"
    for tier in TIERS:
        values = results[tier][k]
        rmse = [v[0] for v in values]
        row += f"{len(values):>4}{np.median(rmse):>9.3f}{min(rmse):>8.3f}-{max(rmse):<7.3f}{sum(v[1] for v in values):>5}"
    print(row)
