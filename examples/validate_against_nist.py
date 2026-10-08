"""Do the computed surfaces put the proton on the right side, by the right amount? Compare their well gaps with the measured ones.

For an asymmetric complex such as [H2O...H...NH3]+ the energy difference between the two wells, with the heavy atoms far apart, is the
difference of the two bases' proton affinities: ammonia holds a proton 162.6 kJ/mol (1.69 eV) more tightly than water. For the anions it is the
difference of the gas-phase acidities (methoxide against fluoride, fluoride against chloride). Those are measured, and `substrate.datasets`
holds them with their provenance (NIST Chemistry WebBook). This script puts each asymmetric reference's computed gap beside the measured one:

    1. from the scans already computed: the gap at each heavy-atom distance of the reference grid (it is still approaching its limit there)
    2. from extra single points with the heavy atoms 6, 8 and 10 A apart: the proton placed at the lowest point near each end, and the gap
       extrapolated to infinite separation from its 1/R^2 dependence (the charge-dipole and charge-induced-dipole terms)
    3. the same gap with each of the four separated fragments (each base and its protonated form, or each anion and its acid) RELAXED: the
       rigid scans hold every fragment in the geometry it has inside the complex, which is not its own minimum, and that costs energy
       the measured proton affinities do not include
    4. the same, plus the difference between an enthalpy at 298 K and an electronic energy: the measured values are enthalpies, the computed
       ones electronic energies, and the two bases differ in zero-point energy and thermal terms (harmonic, rigid rotor; at B3LYP in the
       molecule's own basis, and applied to all three methods, whose Hessians are not all available). This is the like-for-like comparison.

What is still NOT in the comparison, and so limits how close the two can be: the methods and basis sets are the study's (the fluoride-methanol
basis dependence is worked out in docs/findings.md), the thermal terms are harmonic (soft torsions are not), and the measured values carry
uncertainties of their own, which `datasets.py` reports where NIST gives them.

    python examples/validate_against_nist.py            # reads the cached scans; the large-distance points are computed once (minutes)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate import Pipeline, default_registry, load_experiment
from substrate.datasets import TEMPLATE_BASES, experimental_well_gap
from substrate.diatomics import DIATOMICS
from substrate.engines.qc_scan import build_geometry
from substrate.fragments import exchange, separated_fragments
from substrate.molecules import resolve_molecule
from substrate.pes import prominent_minima
from substrate.qc import PySCFProgram, QCCache, QCJob, compute_cached

REFERENCES = Path(__file__).resolve().parent.parent / "experiments" / "references"
DISTANCES = (6.0, 8.0, 10.0)                              # angstrom between the two heavy atoms
OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)     # proton-heavy-atom distance relative to the free diatomic's equilibrium length
HARTREE_EV = 27.211386245988
KJ_PER_EV = 96.48533212331002

parser = argparse.ArgumentParser(description="Compare the computed well gaps of the asymmetric references with the measured proton-affinity differences")
parser.add_argument("--json", metavar="PATH", help="also write the numbers as JSON")
arguments = parser.parse_args()

pipeline = Pipeline(default_registry())
program = PySCFProgram()
if not program.available():
    sys.exit("PySCF is not available. On Windows, run scripts/setup_qc_env.sh in WSL (see the README).")


def lowest_near_end(energies: np.ndarray, positions: np.ndarray) -> float:
    """The minimum of a one-end scan, refined by a parabola through the lowest point and its neighbours (the well is stiff: a 0.05 A step alone is 14 meV)."""
    i = int(np.argmin(energies))
    if 0 < i < len(energies) - 1:
        a, b, c = np.polyfit(positions[i - 1:i + 2], energies[i - 1:i + 2], 2)
        if a > 0:
            return float(np.polyval((a, b, c), -b / (2 * a)))
    return float(energies[i])


def asymptotic_gaps(molecule: dict, theory: str, basis: str) -> dict[float, float]:
    """{R: E(proton at the donor end) - E(proton at the acceptor end)} in eV, one parabola-refined minimum per end."""
    charge, spin = int(molecule.get("charge", 0)), int(molecule.get("spin", 0))
    atoms = molecule["atoms"]
    donor, acceptor = atoms[molecule["donor"]][0], atoms[molecule["acceptor"]][0]
    jobs, layout = [], {}
    for r in DISTANCES:
        for end, element, sign in (("donor", donor, -1.0), ("acceptor", acceptor, +1.0)):
            lengths = np.array([DIATOMICS[element].r_e + d for d in OFFSETS])
            x = sign * (r / 2.0 - lengths)                 # the proton lies between the atoms, `length` from its own end
            layout[(r, end)] = (len(jobs), lengths)
            jobs += [QCJob(build_geometry(molecule, float(xi), r), charge, spin, theory, basis) for xi in x]
    results, n_run, _ = compute_cached(program, jobs, QCCache())
    if any(not res.converged for res in results):
        raise SystemExit("an SCF did not converge for the large-distance points")
    energy = np.array([res.energy_hartree for res in results]) * HARTREE_EV
    out = {}
    for r in DISTANCES:
        ends = {}
        for end in ("donor", "acceptor"):
            start, lengths = layout[(r, end)]
            ends[end] = lowest_near_end(energy[start:start + len(lengths)], lengths)
        out[r] = ends["donor"] - ends["acceptor"]
    return out


def fragment_results(molecule: dict, theory: str, basis: str, task: str) -> dict:
    """{(end, holds the proton): QCResult} for the four separated fragments (see `substrate.fragments`), each run with `task`."""
    fragments = separated_fragments(molecule)
    results, _, _ = compute_cached(program, [QCJob(atoms, charge, 0, theory, basis, task) for atoms, charge in fragments.values()], QCCache())
    if any(not res.converged for res in results):
        raise SystemExit(f"a fragment {task} did not converge")
    return dict(zip(fragments, results))


def relaxed_gap(molecule: dict, theory: str, basis: str) -> float:
    """The gap in eV between the two protonation states with every separated fragment relaxed."""
    return exchange({key: res.energy_hartree for key, res in fragment_results(molecule, theory, basis, "relax").items()}) * HARTREE_EV


THERMAL_SHIFTS: dict[tuple[str, str], float] = {}


def thermal_shift(molecule: dict, basis: str, template: str) -> float:
    """What the harmonic enthalpy at 298 K adds to the relaxed gap, in eV: the exchange's enthalpy minus its electronic energy, at B3LYP (the
    Hessian is analytic there), each fragment at its own minimum. The same shift is applied to every method at this molecule and basis."""
    key = (template, basis)
    if key not in THERMAL_SHIFTS:
        fragments = fragment_results(molecule, "dft:b3lyp", basis, "thermo")
        THERMAL_SHIFTS[key] = (exchange({k: r.enthalpy_298_hartree for k, r in fragments.items()})
                               - exchange({k: r.energy_hartree for k, r in fragments.items()})) * HARTREE_EV
    return THERMAL_SHIFTS[key]


def extrapolate(gaps: dict[float, float]) -> float:
    """The limit of gap(R) = g_inf + c / R^2, from the two largest distances."""
    (r1, g1), (r2, g2) = sorted(gaps.items())[-2:]
    c = (g1 - g2) / (1.0 / r1**2 - 1.0 / r2**2)
    return g2 - c / r2**2


rows = []
for path in sorted(REFERENCES.glob("*.yaml")):
    experiment = load_experiment(path)
    molecule = resolve_molecule(experiment.system.structure["molecule"])
    template = molecule["template"]
    if template not in TEMPLATE_BASES or TEMPLATE_BASES[template][0] == TEMPLATE_BASES[template][1]:
        continue                                           # a symmetric complex: the two wells are equal by construction
    solved = pipeline.run(experiment.system, experiment.propagation).final
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    scan_gaps = {}
    for k, rj in enumerate(r):
        wells = prominent_minima(e[:, k])
        if len(wells) == 2:
            scan_gaps[float(rj)] = float(e[wells[0], k] - e[wells[1], k])
    r_max = max(scan_gaps)
    method = solved.structure["method"]
    gaps = asymptotic_gaps(solved.structure["molecule"], method["theory"], method["basis"])
    limit = extrapolate(gaps)
    relaxed = relaxed_gap(solved.structure["molecule"], method["theory"], method["basis"])
    shift = thermal_shift(solved.structure["molecule"], method["basis"], template)
    measured = experimental_well_gap(template)
    rows.append({"reference": path.stem, "template": template, "theory": method["theory"], "basis": method["basis"], "scan_r_max": r_max,
                 "scan_gap_at_r_max": scan_gaps[r_max], "gaps": {str(k): v for k, v in gaps.items()}, "extrapolated": limit, "relaxed": relaxed,
                 "thermal_shift": shift, "with_thermal": relaxed + shift, "measured": measured.value, "measured_sigma": measured.sigma,
                 "source": measured.source})

print("Computed well gaps (donor-side well minus acceptor-side well, eV) from the rigid scans, against the measured proton-affinity difference\n")
print(f"{'reference':<28}{'scan R_max':>11}{'gap at it':>10}" + "".join(f"{f'R = {r:g}':>9}" for r in DISTANCES) + f"{'R -> inf':>10}{'measured':>10}{'rigid err':>10}")
for row in rows:
    print(f"{row['reference']:<28}{row['scan_r_max']:>11.2f}{row['scan_gap_at_r_max']:>10.3f}" + "".join(f"{row['gaps'][str(d)]:>9.3f}" for d in DISTANCES)
          + f"{row['extrapolated']:>10.3f}{row['measured']:>10.3f}{row['extrapolated'] - row['measured']:>+10.3f}")

print("\nWith each fragment relaxed, and then with the enthalpy at 298 K (B3LYP harmonic thermal shift), eV\n")
print(f"{'reference':<28}{'relaxed':>9}{'thermal':>9}{'+ thermal':>10}{'measured':>10}{'+-':>7}{'relaxed err':>12}{'+ thermal err':>14}")
for row in rows:
    sigma = row["measured_sigma"]
    print(f"{row['reference']:<28}{row['relaxed']:>9.3f}{row['thermal_shift']:>+9.3f}{row['with_thermal']:>10.3f}{row['measured']:>10.3f}"
          f"{'-' if sigma is None else f'{sigma:.3f}':>7}{row['relaxed'] - row['measured']:>+12.3f}{row['with_thermal'] - row['measured']:>+14.3f}")

print("\nby method: the computed gap minus the measured one (eV)\n")
for theory in sorted({row["theory"] for row in rows}):
    for label, key in (("rigid", "extrapolated"), ("relaxed", "relaxed"), ("+ thermal", "with_thermal")):
        errors = [row[key] - row["measured"] for row in rows if row["theory"] == theory]
        print(f"  {theory:<12} {label:<10} mean {np.mean(errors):+.3f}, mean |error| {np.mean(np.abs(errors)):.3f}, worst {max(errors, key=abs):+.3f}   ({len(errors)} ions)")
if arguments.json:
    import json
    Path(arguments.json).write_text(json.dumps(rows, indent=1))
    print(f"\nwritten to {arguments.json}")
