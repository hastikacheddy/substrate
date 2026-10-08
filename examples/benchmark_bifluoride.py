"""How wrong are the study's methods? Bifluoride FHF-, HF / B3LYP / MP2 and CCSD(T), in the study's basis and in aug-cc-pVTZ.

The transfer study compares surfaces from three methods and shows they differ by a lot (the barrier at F...F = 2.74 A is 0.63 eV at HF and 0.25 eV
at B3LYP). That says nothing about which is right. This script puts coupled cluster, CCSD(T), beside them on the SAME rigid scan grid
(experiments/benchmarks/: six distances, 21 proton positions, all-electron), at two basis sets, so that the method error and the basis error can be
read separately:

    method error   a method against CCSD(T) in the same basis      (6-31+G* against CCSD(T)/6-31+G*; aug-cc-pVTZ against CCSD(T)/aug-cc-pVTZ)
    basis error    one method in the two basis sets
    total          the study's own surfaces (6-31+G*) against CCSD(T)/aug-cc-pVTZ, the best surface here

    python examples/benchmark_bifluoride.py              # reads the energies from the cache (compute them first: see below)
    python examples/benchmark_bifluoride.py --figure docs/images/benchmark-barrier   # also draws the barrier figure (needs matplotlib)

The CCSD(T)/aug-cc-pVTZ scan is 67 energies at about 90 s each on this machine (one worker, all cores); every other surface is minutes. Run each
file in experiments/benchmarks/ once through `python -m substrate run` (or this script) to fill the cache.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, default_registry, load_experiment
from substrate.calibration import FitSettings
from substrate.pes import locate_wells
from substrate.transfer import cross_predict, make_reference

parser = argparse.ArgumentParser(description="Bifluoride FHF-: HF, B3LYP, MP2 and CCSD(T), in two basis sets, on one rigid scan grid")
parser.add_argument("--figure", metavar="PREFIX", help="write PREFIX-light.png and PREFIX-dark.png: the barrier against F...F distance, by method and basis")
arguments = parser.parse_args()

BENCHMARKS = Path(__file__).resolve().parent.parent / "experiments" / "benchmarks"
THEORIES = [("hf", "HF"), ("b3lyp", "B3LYP"), ("mp2", "MP2"), ("ccsdt", "CCSD(T)")]
BASES = [("631pg", "6-31+G*"), ("augccpvtz", "aug-cc-pVTZ")]
GOLD = ("ccsdt", "augccpvtz")
WINDOW = 1.5                                   # eV above each surface's minimum: where the nuclei are

pipeline = Pipeline(default_registry())
surfaces, systems = {}, {}
missing = []
for basis_tag, _ in BASES:
    for tag, _ in THEORIES:
        path = BENCHMARKS / f"fhf_{tag}_{basis_tag}.yaml"
        experiment = load_experiment(path)
        solved = pipeline.run(experiment.system, experiment.propagation).final
        if solved.obs("calculations_run") > 0:
            missing.append(f"{tag}/{basis_tag} ({solved.obs('calculations_run'):.0f} energies were not cached)")
        surfaces[(tag, basis_tag)], systems[(tag, basis_tag)] = solved, experiment.system
if missing:
    print("WARNING: these were computed now, not read: " + "; ".join(missing) + "\n")

x = surfaces[GOLD].obs("surface_x")
r = surfaces[GOLD].obs("surface_r")
energy = {key: s.obs("surface_energy") for key, s in surfaces.items()}          # eV above each surface's own minimum
label = {(t, b): f"{tl}/{bl}" for t, tl in THEORIES for b, bl in BASES}


def barrier(column: np.ndarray) -> float:
    wells = locate_wells(column)
    return float(column[wells[1]] - column[wells[0]]) if wells else 0.0


def well_position(column: np.ndarray) -> float:
    return float(abs(x[np.argmin(column)]))


def surface_error(model: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """rmse and worst error (eV) of `model` against `reference` over the reference's points within WINDOW eV, after the best constant shift."""
    inside = reference <= WINDOW
    shift = float(np.mean(reference[inside] - model[inside]))
    error = (model + shift - reference)[inside]
    return float(np.sqrt(np.mean(error**2))), float(np.abs(error).max())


print("1. The proton-transfer barrier (eV; top of the barrier above the well, 0 = a single well) at each F...F distance\n")
print(f"   {'':<22}" + "".join(f"{d:>8.2f}" for d in r))
for basis_tag, _ in BASES:
    for tag, _ in THEORIES:
        print(f"   {label[(tag, basis_tag)]:<22}" + "".join(f"{barrier(energy[(tag, basis_tag)][:, k]):>8.3f}" for k in range(len(r))))
print("\n   where the proton sits (|x| of the lowest point, angstrom from the centre; 0 = centred)\n")
print(f"   {'':<22}" + "".join(f"{d:>8.2f}" for d in r))
for basis_tag, _ in BASES:
    for tag, _ in THEORIES:
        print(f"   {label[(tag, basis_tag)]:<22}" + "".join(f"{well_position(energy[(tag, basis_tag)][:, k]):>8.2f}" for k in range(len(r))))

gold = energy[GOLD]
print(f"\n2. Error against CCSD(T)/aug-cc-pVTZ over the points within {WINDOW:g} eV of its minimum (best constant shift): rmse / worst (eV), and the barrier error at each distance\n")
print(f"   {'':<22}{'rmse':>8}{'worst':>8}  barrier error at " + " ".join(f"{d:.2f}" for d in r))
for basis_tag, _ in BASES:
    for tag, _ in THEORIES:
        if (tag, basis_tag) == GOLD:
            continue
        rmse, worst = surface_error(energy[(tag, basis_tag)], gold)
        errors = [barrier(energy[(tag, basis_tag)][:, k]) - barrier(gold[:, k]) for k in range(len(r))]
        print(f"   {label[(tag, basis_tag)]:<22}{rmse:>8.3f}{worst:>8.3f}  " + " ".join(f"{e:>+5.2f}" for e in errors))

print("\n3. Method error and basis error separately (rmse over the points within the window of the reference, eV)\n")
print(f"   {'':<10}{'method error':>16}{'method error':>16}{'basis error':>16}")
print(f"   {'':<10}{'in 6-31+G*':>16}{'in aug-cc-pVTZ':>16}{'(its own two)':>16}")
for tag, tl in THEORIES:
    cells = []
    for basis_tag, _ in BASES:
        cells.append("-" if tag == "ccsdt" else f"{surface_error(energy[(tag, basis_tag)], energy[('ccsdt', basis_tag)])[0]:.3f}")
    basis = surface_error(energy[(tag, "631pg")], energy[(tag, "augccpvtz")])[0]
    print(f"   {tl:<10}{cells[0]:>16}{cells[1]:>16}{basis:>16.3f}")

print("\n4. The model fitted to each surface (shared Morse, a 1.5 eV window, the coupling defined at 2.7 A), and each fit applied unchanged to the gold-standard surface\n")
settings = FitSettings(reference_distance=2.7, window_ev=WINDOW)
references = {key: make_reference(label[key], systems[key], surfaces[key], settings) for key in surfaces}
gold_reference = references[GOLD]
print(f"   {'fitted to':<22}{'own rmse':>9}{'on CCSD(T)/aug-cc-pVTZ':>24}{'its own surface on it':>24}")
for basis_tag, _ in BASES:
    for tag, _ in THEORIES:
        reference = references[(tag, basis_tag)]
        own = cross_predict(reference.calibration, reference).rmse_window
        onto_gold = cross_predict(reference.calibration, gold_reference).rmse_window
        raw = surface_error(energy[(tag, basis_tag)], gold)[0]
        print(f"   {label[(tag, basis_tag)]:<22}{own:>9.4f}{onto_gold:>24.3f}{raw:>24.3f}")
print("\n   (the last column is the raw surface against the gold standard, as in table 2: a fit adds its own error to the method's)")


# -- the figure ---------------------------------------------------------------------------------------------------------------------------------------------
def draw_figure(prefix: str) -> None:
    """Barrier against distance in each basis, one panel per basis: the three approximate methods against CCSD(T), which is the reference."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("--figure needs matplotlib (pip install matplotlib)")
    themes = {
        "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e4e3df", series=["#2a78d6", "#eb6834", "#1baf7a"]),
        "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7", grid="#34332f", series=["#3987e5", "#d95926", "#199e70"]),
    }
    approximate = [("hf", "HF", "o"), ("b3lyp", "B3LYP", "s"), ("mp2", "MP2", "^")]
    for name, theme in themes.items():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, dpi=190)
        fig.patch.set_facecolor(theme["surface"])
        for ax, (basis_tag, basis_label) in zip(axes, [("631pg", "6-31+G*  (the study's basis)"), ("augccpvtz", "aug-cc-pVTZ  (large basis)")]):
            ax.set_facecolor(theme["surface"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(theme["muted"])
            ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=theme["muted"], labelsize=9)
            curves = []                                                       # (barrier at the last distance, label, colour) for direct labelling
            if basis_tag != "augccpvtz":                                      # the reference of the other panel, for the basis effect
                reference = [barrier(energy[("ccsdt", "augccpvtz")][:, k]) for k in range(len(r))]
                ax.plot(r, reference, color=theme["muted"], linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
                curves.append((reference[-1], "CCSD(T)/aug-cc-pVTZ", theme["muted"]))
            for (tag, label, marker), colour in zip(approximate, theme["series"]):
                y = [barrier(energy[(tag, basis_tag)][:, k]) for k in range(len(r))]
                ax.plot(r, y, color=colour, linewidth=2.0, marker=marker, markersize=6, markeredgecolor=theme["surface"], markeredgewidth=1.2, zorder=3)
                curves.append((y[-1], label, colour))
            y = [barrier(energy[("ccsdt", basis_tag)][:, k]) for k in range(len(r))]
            ax.plot(r, y, color=theme["ink"], linewidth=2.6, marker="D", markersize=6, markeredgecolor=theme["surface"], markeredgewidth=1.2, zorder=4)
            curves.append((y[-1], "CCSD(T)", theme["ink"]))
            curves.sort()
            placed = []
            for value, label, colour in curves:                              # keep the direct labels from colliding
                position = value if not placed else max(value, placed[-1] + 0.075)
                placed.append(position)
                ax.annotate(label, (r[-1], value), xytext=(r[-1] + 0.025, position), color=colour if colour != theme["ink"] else theme["ink"], fontsize=9,
                            va="center", annotation_clip=False, fontweight="bold" if label == "CCSD(T)" else "normal")
            ax.set_xlim(r[0], r[-1] + 0.2)
            ax.set_ylim(-0.03, 1.32)
            ax.set_xticks(r)
            ax.set_title(basis_label, color=theme["ink"], fontsize=11, loc="left", pad=8)
            ax.set_xlabel("F···F distance (Å)", color=theme["muted"], fontsize=10)
        axes[0].set_ylabel("proton-transfer barrier (eV)", color=theme["muted"], fontsize=10)
        fig.suptitle("Bifluoride FHF⁻: the barrier, by method and basis", color=theme["ink"], fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
        fig.text(0.012, 0.915, "Rigid scan, all electrons. HF puts the barrier too high and B3LYP too low in both bases; MP2 follows coupled cluster.", color=theme["muted"],
                 fontsize=9.5, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        out = f"{prefix}-{name}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")


if arguments.figure:
    draw_figure(arguments.figure)
