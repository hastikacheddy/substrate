"""Is the fluoride-methanol well gap a basis-set problem? Take the one asymmetric ion whose computed gap is far from the measured one.

`validate_against_nist.py` finds the long-range gap of [F...H...OCH3]- in the study's 6-31+G* basis well above the measured one (the
measured value is 0.435 +- 0.098 eV, from the gas-phase acidities of HF and methanol), where the other asymmetric ions are within about 0.1 eV. This script
asks how much of that is the basis and how much the method, on the four separated fragments (F-, HF, CH3O-, CH3OH):

    1. B3LYP with each fragment relaxed, in 6-31+G*, aug-cc-pVDZ and aug-cc-pVTZ: the basis alone
    2. MP2 and CCSD(T) in aug-cc-pVTZ at the B3LYP/aug-cc-pVTZ geometries: the method, in the largest basis
    3. the harmonic enthalpy correction (B3LYP, in 6-31+G* and in aug-cc-pVDZ, to show it hardly depends on the basis), added to the
       CCSD(T) electronic gap: the like-for-like comparison with the measured enthalpy difference

The coupled-cluster energies are a few minutes to an hour each, so the first run is long; everything is cached on disk and a repeat is instant.

    python examples/basis_check_fluoride_methanol.py [--json PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from substrate.datasets import experimental_well_gap
from substrate.fragments import exchange, separated_fragments
from substrate.molecules import resolve_molecule
from substrate.qc import PySCFProgram, QCCache, QCJob, compute_cached

HARTREE_EV = 27.211386245988
BASES = ("6-31+G*", "aug-cc-pVDZ", "aug-cc-pVTZ")

parser = argparse.ArgumentParser(description="Basis-set and method dependence of the fluoride-methanol well gap")
parser.add_argument("--json", metavar="PATH", help="also write the numbers as JSON")
arguments = parser.parse_args()

program = PySCFProgram(timeout=4 * 3600.0)                  # a coupled-cluster chunk of four fragments can take over an hour
if not program.available():
    sys.exit("PySCF is not available. On Windows, run scripts/setup_qc_env.sh in WSL (see the README).")
molecule = resolve_molecule({"template": "fluoride_methanol_anion"})
fragments = separated_fragments(molecule)
cache = QCCache()


def run(jobs):
    results, _, _ = compute_cached(program, jobs, cache)
    if any(not res.converged for res in results):
        raise SystemExit("a calculation did not converge")
    return dict(zip(fragments, results))


def gap_ev(results, field="energy_hartree"):
    return exchange({key: getattr(res, field) for key, res in results.items()}) * HARTREE_EV


report = {"relaxed_b3lyp": {}, "thermal_shift": {}}
print("Fluoride-methanol: the gap between the two protonation states of the separated fragments, eV\n")
print("B3LYP, every fragment relaxed")
relaxed = {}
for basis in BASES:
    relaxed[basis] = run([QCJob(atoms, charge, 0, "dft:b3lyp", basis, "relax") for atoms, charge in fragments.values()])
    report["relaxed_b3lyp"][basis] = gap_ev(relaxed[basis])
    print(f"  {basis:<14}{report['relaxed_b3lyp'][basis]:>8.3f}")

best = relaxed["aug-cc-pVTZ"]
print("\nIn aug-cc-pVTZ, at the B3LYP geometries")
report["aug-cc-pVTZ"] = {"b3lyp": report["relaxed_b3lyp"]["aug-cc-pVTZ"]}
print(f"  {'B3LYP':<14}{report['aug-cc-pVTZ']['b3lyp']:>8.3f}")
for theory in ("mp2", "ccsd(t)"):
    results = run([QCJob(best[key].geometry, charge, 0, theory, "aug-cc-pVTZ") for key, (atoms, charge) in fragments.items()])
    report["aug-cc-pVTZ"][theory] = gap_ev(results)
    print(f"  {theory.upper() if theory == 'mp2' else 'CCSD(T)':<14}{report['aug-cc-pVTZ'][theory]:>8.3f}")

print("\nEnthalpy at 298 K minus electronic energy (harmonic, B3LYP), eV")
for basis in ("6-31+G*", "aug-cc-pVDZ"):
    thermo = run([QCJob(atoms, charge, 0, "dft:b3lyp", basis, "thermo") for atoms, charge in fragments.values()])
    report["thermal_shift"][basis] = gap_ev(thermo, "enthalpy_298_hartree") - gap_ev(thermo)
    print(f"  {basis:<14}{report['thermal_shift'][basis]:>+8.3f}")

measured = experimental_well_gap("fluoride_methanol_anion")
shift = report["thermal_shift"]["aug-cc-pVDZ"]
final = report["aug-cc-pVTZ"]["ccsd(t)"] + shift
report["ccsd(t)_with_thermal"] = final
report["measured"], report["measured_sigma"] = measured.value, measured.sigma
print(f"\nCCSD(T)/aug-cc-pVTZ + thermal {final:.3f} eV, measured {measured.value:.3f} +- {measured.sigma:.3f} eV (difference {final - measured.value:+.3f}, "
      f"{(final - measured.value) / measured.sigma:+.1f} sigma)")
if arguments.json:
    import json
    Path(arguments.json).write_text(json.dumps(report, indent=1))
    print(f"\nwritten to {arguments.json}")
