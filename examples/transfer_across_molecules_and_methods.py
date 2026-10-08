"""Do model parameters fitted to one reference carry over to another molecule or another level of theory?

The valence-bond model engine was calibrated against one reference (the Zundel cation, Hartree-Fock/6-31G*; see
calibrate_against_real_chemistry.py). Here it is calibrated against twenty-one real surfaces and the fitted parameters are tested on each
other:

    Zundel cation  H5O2+  (O-H-O)   HF, B3LYP, MP2 at 6-31G*, and HF at cc-pVDZ
    ammonium dimer N2H7+  (N-H-N)   HF, B3LYP at 6-31G*
    bifluoride     FHF-   (F-H-F)   HF, B3LYP, MP2 at 6-31+G*
    water-ammonia  H2O-H-NH3+ (O-H-N)  HF, B3LYP, MP2 at 6-31G*   (ASYMMETRIC: the model's energy offset is a free parameter,
                                                       and its experiment files declare a wider fit window, 2.5 eV)
    methanol-water H2O-H-O(H)CH3+ (O-H-O)  HF, B3LYP, MP2 at 6-31G*  (MILDLY asymmetric: proton-affinity gap 0.66 eV (NIST), against 1.69 for water-ammonia)
    ammonia-methylamine H3N-H-NH2CH3+ (N-H-N)  HF, B3LYP, MP2 at 6-31G*  (MILDLY asymmetric, ~0.47 eV: the methyl derivative of the ammonium dimer)
    fluoride-methanol  F-H-OCH3- (F-H-O)  HF, B3LYP, MP2 at 6-31+G*  (an ANION, mildly asymmetric: one bifluoride fluoride swapped for methoxide; not a pure derivative)

The comparisons (none is a statement about the fit's own uncertainty):

  1. each fit, and where it fails;
  2. the fitted parameters side by side: how much they change across methods and across molecules;
  3. cross-prediction: every calibration against every reference, its parameters unchanged (only the energy zero is shifted);
  4. downstream: rates and isotope effects from the real chain against the same chain on each calibration;
  5. few-shot learning: with k of a target's heavy-atom distances computed, how well does a fit predict the others, from scratch and
     with another reference's parameters as a Gaussian prior?

Needs PySCF (WSL on Windows; see scripts/setup_qc_env.sh). The reference energies come from the on-disk cache after a first run; a
first run computes about 1,200 single points. Run from the project root:

    python examples/transfer_across_molecules_and_methods.py [--refs zundel_hf,fhf_hf,...] [--quick] [--only learning]
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from substrate import Pipeline, default_registry, load_experiment
from substrate.calibration import EVB2D_PARAMETERS, FitSettings
from substrate.qc import PySCFProgram
from substrate.transfer import (
    fit_settings, format_baselines, format_matrix, format_own_fits, learning_curve, make_reference, parameter_table, rate_comparison, spread_ratio, transfer_matrix,
)

ALL = ["zundel_hf", "zundel_b3lyp", "zundel_mp2", "zundel_hf_ccpvdz", "n2h7_hf", "n2h7_b3lyp", "fhf_hf", "fhf_b3lyp", "fhf_mp2",
       "water_ammonia_hf", "water_ammonia_b3lyp", "water_ammonia_mp2",
       "methanol_water_hf", "methanol_water_b3lyp", "methanol_water_mp2",
       "ammonia_methylamine_hf", "ammonia_methylamine_b3lyp", "ammonia_methylamine_mp2",
       "fluoride_methanol_hf", "fluoride_methanol_b3lyp", "fluoride_methanol_mp2"]
COMMON_REFERENCE_DISTANCE = 2.7                    # one definition of the coupling for every fit, so parameters share variables
PRIOR_SIGMAS = (0.1, 0.3, 1.0)                     # relative prior widths swept in the learning curves (none is tuned on the test set)
COUNTS = (0, 1, 2, 3, 6)
#: the symmetric parent of each one-methyl derivative (by template name): a natural prior for it
PARENTS = {"methanol_water_cation": "zundel_cation", "ammonia_methylamine_cation": "ammonium_dimer_cation",
           "fluoride_methanol_anion": "bifluoride_anion"}               # the last is a nominal parent: a fluoride is swapped for methoxide

parser = argparse.ArgumentParser()
parser.add_argument("--refs", default=",".join(ALL))
parser.add_argument("--quick", action="store_true", help="skip the rate comparison and learning curves")
parser.add_argument("--only", choices=["learning"], help="build the references, then run only the few-shot learning section")
args = parser.parse_args()
names = args.refs.split(",")

if not PySCFProgram().available():
    sys.exit("PySCF is not available. On Windows, run scripts/setup_qc_env.sh in WSL (see the README).")

pipeline = Pipeline(default_registry())
settings = FitSettings(reference_distance=COMMON_REFERENCE_DISTANCE)

print("0. The reference surfaces (rigid-group scans, 21 proton positions x 11 heavy-atom distances, mirror-symmetric)\n")
references = {}
for name in names:
    started = time.time()
    experiment = load_experiment(ROOT / "experiments" / "references" / f"{name}.yaml")
    solved = pipeline.run(experiment.system, experiment.propagation).final
    references[name] = make_reference(name, experiment.system, solved, fit_settings(experiment, settings))
    print(f"   {name:<18}{solved.obs('calculations_run'):>5.0f} computed, {solved.obs('calculations_cached'):>4.0f} cached   "
          f"{time.time() - started:6.1f} s")

def learning_section():
    print("\n5. Few-shot learning: fit on k of the target's heavy-atom distances (0 = the source's parameters alone), score on the held-out ones\n")
    print("   Training distances are the even-numbered ones of 11, the test set the five odd-numbered ones (never trained on).")
    print("   Source rule, applied to every target: the same molecule's Hartree-Fock reference if the target is another method; for contrast,")
    print("   another molecule's Hartree-Fock reference (Zundel's, or the ammonium dimer's for Zundel targets); and for the one-methyl derivatives")
    print("   also their symmetric parent (Zundel for methanol-water, the ammonium dimer for ammonia-methylamine)." + chr(92) + "n")
    print("   held-out rmse (eV) | 'scratch' = no prior; 'prior s' = prior with relative sigma s")
    for target_name, target in references.items():
        sources = []
        same = next((n for n in names if references[n].molecule == target.molecule and references[n].method.startswith("hf/") and n != target_name), None)
        if same:
            sources.append(("same molecule", same))
        other_molecule = "zundel_cation" if target.molecule != "zundel_cation" else "ammonium_dimer_cation"
        other = next((n for n in names if references[n].molecule == other_molecule and references[n].method.startswith("hf/")), None)
        if other and other != target_name:
            sources.append(("other molecule", other))
        parent_template = PARENTS.get(target.molecule)
        parent = next((n for n in names if parent_template and references[n].molecule == parent_template and references[n].method in ("hf/6-31g*", "hf/6-31+g*")), None)
        if parent and sources and sources[-1][1] == parent:
            sources[-1] = ("symmetric parent", parent)                             # the contrast source already is the parent
        elif parent:
            sources.append(("symmetric parent", parent))
        tset = replace(settings, window_ev=target.window_ev)                       # the target's own fit window
        own_floor = learning_curve(target, counts=(6,), settings=tset)[0].rmse_ev
        scratch = learning_curve(target, counts=COUNTS[1:], settings=tset)
        print(f"\n   target {target_name}   (best this model form does with 6 training distances, no prior: {own_floor:.4f})")
        print(f"   {'source':<28}{'prior':>8}" + "".join(f"{f'k={k}':>9}" for k in COUNTS))
        print(f"   {'(none)':<28}{'scratch':>8}{'-':>9}" + "".join(f"{p.rmse_ev:>9.4f}" for p in scratch))
        for kind, source_name in sources:
            for sigma in PRIOR_SIGMAS:
                curve = learning_curve(target, counts=COUNTS, source=references[source_name].calibration, relative_sigma=sigma, settings=tset)
                print(f"   {f'{source_name} ({kind})':<28}{f's={sigma:g}':>8}" + "".join(f"{p.rmse_ev:>9.4f}" for p in curve))
    print("\nThe prior is the source's fitted values with a Gaussian of relative width s. At k=0 the source's parameters are scored with the best"
          " constant energy shift on the held-out points, which flatters them.")


if args.only == "learning":
    learning_section()
    sys.exit(0)

matrix = transfer_matrix(references)
print("\n1. Each calibration against its own reference\n")
print(format_own_fits(references, matrix))

print("\n2. The fitted parameters (value, with the fit's sigma where it has one)\n")
table = parameter_table(references)
width = max(17, max(len(n) for n in names) + 2)
print(f"{'':<17}" + "".join(f"{n:>{width}}" for n in names))
for parameter, row in table.items():
    cells = ["-" if row[n][0] is None else f"{row[n][0]:.3g}" + (f" +/-{row[n][1]:.2g}" if row[n][1] is not None else " (bound)") for n in names]
    print(f"{parameter:<17}" + "".join(f"{c:>{width}}" for c in cells))
molecules = sorted({ref.molecule for ref in references.values()})
print("\n   largest / smallest value of each parameter:")
print(f"   {'':<16}" + "".join(f"{m.split('_')[0]:>14}" for m in molecules) + f"{'across molecules':>20}")
for parameter, row in table.items():
    if parameter == "diabatic_offset":
        continue                                                                  # a signed energy: a ratio of offsets means nothing
    by_molecule = [spread_ratio([row[n][0] for n, ref in references.items() if ref.molecule == m]) for m in molecules]
    hf = [row[n][0] for n, ref in references.items() if ref.method.startswith("hf/")]
    fmt_ratio = lambda v, w: f"{'n/a':>{w}}" if v != v else f"{v:>{w}.2f}"
    print(f"   {parameter:<16}" + "".join(fmt_ratio(v, 14) for v in by_molecule) + fmt_ratio(spread_ratio(hf), 20))
print("   (columns: across the methods computed for that molecule; last: across the molecules' Hartree-Fock references)")

print("\n3. Cross-prediction: row = whose parameters, column = whose real energies; only the energy zero is shifted\n")
print("   rmse (eV) over the target's points up to its fit window above its minimum (the last row of the baselines)\n")
print(format_matrix(matrix, names, "rmse"))
print(format_baselines(references))
print("\n   mean |barrier error| (eV) at the target's double-well distances\n")
print(format_matrix(matrix, names, "barrier"))
print("\n   target distances where the model has the wrong number of wells (of 11)\n")
print(format_matrix(matrix, names, "wells"))

pinned = {n: sorted(ref.calibration.pinned) for n, ref in references.items() if ref.calibration.pinned}
if pinned:
    print("\n   Some fits sit on a bound (see 'pinned' in section 1): " + "; ".join(f"{n}: {', '.join(p)}" for n, p in pinned.items()))
    print("   Do the conclusions depend on it? Refit every reference with that bound moved out (oo_equilibrium 1.8-10 A, was 1.8-4) and compare:")
    wide = FitSettings(reference_distance=COMMON_REFERENCE_DISTANCE, bounds=(("oo_equilibrium", 1.8, 10.0),))
    wide_refs = {n: make_reference(n, ref.system, pipeline.run(ref.system, [ref.system.scale]).final, replace(wide, window_ev=ref.window_ev))
                 for n, ref in references.items()}
    wide_matrix = transfer_matrix(wide_refs)
    drmse = np.array([abs(wide_matrix[k].rmse_window - matrix[k].rmse_window) for k in matrix])
    dbarrier = np.array([abs(wide_matrix[k].barrier_error_mean_abs - matrix[k].barrier_error_mean_abs) for k in matrix])
    print(f"   every cross-prediction rmse changes by at most {drmse.max():.3f} eV (median {np.median(drmse):.3f}); mean barrier errors by at most "
          f"{np.nanmax(dbarrier):.3f} eV (median {np.nanmedian(dbarrier):.3f}); fit rmse by at most "
          f"{max(abs(wide_matrix[(n, n)].rmse_window - matrix[(n, n)].rmse_window) for n in names):.4f} eV")
    print("   (the O...O parameters are not identified by the data: with the bound out, oo_equilibrium and oo_depth wander along a flat direction)")

if args.quick:
    sys.exit(0)

print("\n4. Downstream: k_H and the isotope effect through the real chain (quantum route, 300 K) against the same chain on each calibration\n")
started = time.time()
rows = rate_comparison(references, references, pipeline, progress=lambda m: print(m, flush=True))
print(f"   ({time.time() - started:.0f} s)")
levels = sorted({row.barrier_ev for row in rows})
for level in levels:
    print(f"\n   target barrier {level:g} eV: model rate / real rate  (row = whose parameters, column = whose real chain)")
    print(f"   {'':<17}" + "".join(f"{n:>{max(len(n) + 1, 10)}}" for n in names))
    by = {(r.source, r.target): r for r in rows if r.barrier_ev == level}
    real = {t: next((r for r in rows if r.target == t and r.barrier_ev == level), None) for t in names}
    for s in names:
        cells = []
        for t in names:
            r = by.get((s, t))
            cells.append("-" if r is None or r.ratio is None else (f"{r.ratio:.2f}" if 0.01 < r.ratio < 100 else f"{r.ratio:.0e}"))
        print(f"   {s:<17}" + "".join(f"{c:>{max(len(t) + 1, 10)}}" for c, t in zip(cells, names)))
    print(f"   {'real k_H (1/s)':<17}" + "".join(
        f"{('-' if real[t] is None or not real[t].k_real else f'{real[t].k_real:.1e}'):>{max(len(t) + 1, 10)}}" for t in names))
    print(f"   {'real KIE':<17}" + "".join(
        f"{('-' if real[t] is None or not real[t].kie_real else f'{real[t].kie_real:.0f}'):>{max(len(t) + 1, 10)}}" for t in names))
    print(f"   {'at R (A)':<17}" + "".join(
        f"{('-' if real[t] is None or real[t].distance != real[t].distance else f'{real[t].distance:.2f}'):>{max(len(t) + 1, 10)}}" for t in names))
notes = sorted({(r.target, r.source, r.barrier_ev, r.note) for r in rows if r.note})
if notes:
    print("\n   chains that refused:")
    for target, source, level, note in notes[:20]:
        print(f"     target {target}, source {source}, barrier {level:g}: {note}")

learning_section()
