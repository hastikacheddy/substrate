"""Standalone PySCF worker.

Runs wherever PySCF is installed, including a Linux environment reached through WSL, so it must not import anything from
`substrate`. Protocol: a JSON object {"jobs": [...]} on stdin; one JSON line per result on stdout, then {"done": true}.

    job:    {"id", "atoms": [[symbol, x, y, z], ...] (angstrom), "charge", "spin" (2S), "theory", "basis", "task"}
    result: {"id", "energy" (hartree), "converged", "homo_lumo_gap" (hartree), "seconds"}   or   {"id", "error"}

task is "energy" (the default: one single point), "relax" or "thermo". "relax" minimises the energy over every atom's position from the given
geometry (BFGS on the analytic gradient, Hartree-Fock, MP2 or DFT, closed shell), and returns the relaxed "geometry", the "energy" there and the
"initial_energy" at the geometry it started from; "converged" is then the optimiser's convergence as well as the SCF's. "thermo" relaxes more
tightly (Hartree-Fock or DFT only: the Hessian is analytic), takes the harmonic frequencies at the minimum and returns also the "zero_point"
energy and the total "enthalpy_298" (hartree; electronic energy + zero point + the translational, rotational and vibrational terms at 298.15 K
and 1 atm, rigid rotor and harmonic oscillator); it is reported as not converged if any frequency is imaginary.

theory is "hf", "mp2", "ccsd", "ccsd(t)" or "dft:<functional>" (e.g. "dft:b3lyp"); open-shell systems (spin > 0) use the unrestricted
variants of Hartree-Fock, MP2 and DFT. Coupled cluster is closed-shell only (an open-shell job is refused, not approximated). MP2 and
coupled cluster correlate every electron (no frozen core), so the methods differ only in how they treat correlation. A coupled-cluster
energy is reported as not converged unless both the SCF and the CCSD iterations converged.
"""
from __future__ import annotations

import json
import sys
import time


COUPLED_CLUSTER = ("ccsd", "ccsd(t)")


def _orbital_gap(mf) -> float | None:
    """HOMO-LUMO gap (hartree) from the converged mean-field orbitals; None if there is no virtual orbital."""
    import numpy as np

    energies, occupations = np.atleast_2d(mf.mo_energy), np.atleast_2d(mf.mo_occ)
    occupied = energies[occupations > 0]
    virtual = energies[occupations == 0]
    if occupied.size == 0 or virtual.size == 0:
        return None
    return float(virtual.min() - occupied.max())


def _relax(job: dict, started: float, thermo: bool = False) -> dict:
    """Minimise the energy over all atomic positions (Cartesian BFGS, analytic gradients); closed-shell Hartree-Fock, MP2 and DFT. With `thermo`,
    to a tighter gradient and on a finer DFT grid, then the harmonic analysis at the minimum (Hartree-Fock and DFT: the Hessian is analytic)."""
    import numpy as np
    from pyscf import dft, gto, mp, scf
    from scipy.optimize import minimize

    bohr = 0.529177210903
    theory = job["theory"].lower()
    kind = "a thermo job" if thermo else "a relaxation"
    if int(job["spin"]) != 0 or theory.startswith("ccsd") or (thermo and theory == "mp2"):
        allowed = "Hartree-Fock and DFT" if thermo else "Hartree-Fock, MP2 and DFT"
        raise ValueError(f"{kind} is implemented for closed-shell {allowed} only (got '{job['theory']}', spin {job['spin']})")
    symbols = [a[0] for a in job["atoms"]]

    def mean_field(x):
        mol = gto.M(atom=[[s, tuple(x[3 * i:3 * i + 3] * bohr)] for i, s in enumerate(symbols)], charge=int(job["charge"]), spin=0,
                    basis=job["basis"], unit="Angstrom", verbose=0)
        if theory.startswith("dft:"):
            mf = dft.RKS(mol)
            mf.xc = theory[4:]
            if thermo:
                mf.grids.level = 4                       # second derivatives are noisy on the default grid
        elif theory in ("hf", "mp2"):
            mf = scf.RHF(mol)
        else:
            raise ValueError(f"unknown theory '{job['theory']}' (use 'hf', 'mp2' or 'dft:<functional>' for {kind})")
        mf.conv_tol = 1e-11 if thermo else 1e-10
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.kernel()
        return mf

    def energy_and_gradient(x):
        mf = mean_field(x)
        if theory == "mp2":
            post = mp.MP2(mf)
            post.kernel()
            return post.e_tot, post.nuc_grad_method().kernel().ravel(), bool(mf.converged)
        return mf.e_tot, mf.nuc_grad_method().kernel().ravel(), bool(mf.converged)

    start = np.array([c for a in job["atoms"] for c in a[1:]], dtype=float) / bohr
    initial = energy_and_gradient(start)[0]
    scf_ok = [True]

    def objective(x):
        e, g, ok = energy_and_gradient(x)
        scf_ok[0] = scf_ok[0] and ok
        return e, g

    solution = minimize(objective, start, jac=True, method="BFGS", options={"gtol": 1e-4 if thermo else 3e-4, "maxiter": 200 if thermo else 150})
    final = solution.x * bohr
    result = {"id": job["id"], "energy": float(solution.fun), "initial_energy": float(initial), "converged": bool(solution.success and scf_ok[0]),
              "geometry": [[s, *map(float, final[3 * i:3 * i + 3])] for i, s in enumerate(symbols)], "homo_lumo_gap": None}
    if thermo:
        from pyscf.hessian import thermo as pyscf_thermo
        mf = mean_field(solution.x)
        analysis = pyscf_thermo.harmonic_analysis(mf.mol, mf.Hessian().kernel())
        if np.any(np.imag(analysis["freq_au"]) != 0):                      # a saddle point, not a minimum: no thermochemistry
            result["converged"] = False
        else:
            values = pyscf_thermo.thermo(mf, analysis["freq_au"], 298.15, 101325)
            result["zero_point"] = float(values["ZPE"][0])
            result["enthalpy_298"] = float(values["H_tot"][0])
    result["seconds"] = time.perf_counter() - started
    return result


def run_job(job: dict) -> dict:
    from pyscf import dft, gto, scf

    started = time.perf_counter()
    task = job.get("task", "energy")
    if task in ("relax", "thermo"):
        return _relax(job, started, thermo=task == "thermo")
    if task != "energy":
        raise ValueError(f"unknown task '{task}' (use 'energy', 'relax' or 'thermo')")
    mol = gto.M(
        atom=[[sym, (x, y, z)] for sym, x, y, z in job["atoms"]],
        charge=int(job["charge"]), spin=int(job["spin"]), basis=job["basis"], unit="Angstrom", verbose=0,
    )
    theory = job["theory"].lower()
    open_shell = mol.spin != 0
    if theory in COUPLED_CLUSTER and open_shell:
        raise ValueError(f"'{job['theory']}' is implemented for closed-shell systems only (spin {mol.spin})")
    if theory.startswith("dft:"):
        mf = (dft.UKS if open_shell else dft.RKS)(mol)
        mf.xc = theory[4:]
    elif theory in ("hf", "mp2", *COUPLED_CLUSTER):
        mf = (scf.UHF if open_shell else scf.RHF)(mol)
    else:
        raise ValueError(f"unknown theory '{job['theory']}' (use 'hf', 'mp2', 'ccsd', 'ccsd(t)' or 'dft:<functional>')")
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:                                  # second-order SCF is slower but far more robust
        mf = mf.newton()
        mf.kernel()
    energy, converged = mf.e_tot, bool(mf.converged)
    if theory == "mp2" and mf.converged:
        from pyscf import mp
        correlated = mp.UMP2(mf) if open_shell else mp.MP2(mf)
        correlated.kernel()
        energy = correlated.e_tot
    elif theory in COUPLED_CLUSTER and mf.converged:
        from pyscf import cc
        correlated = cc.CCSD(mf)
        correlated.conv_tol = 1e-9
        correlated.kernel()
        converged = bool(correlated.converged)
        energy = correlated.e_tot
        if theory == "ccsd(t)" and converged:
            energy += correlated.ccsd_t()                    # the perturbative triples correction
    return {
        "id": job["id"], "energy": float(energy), "converged": converged,
        "homo_lumo_gap": _orbital_gap(mf), "seconds": time.perf_counter() - started,
    }


def main() -> None:
    request = json.loads(sys.stdin.read())
    import pyscf

    for job in request["jobs"]:
        try:
            result = run_job(job)
        except Exception as error:                        # report per job; one bad geometry must not lose the rest
            result = {"id": job["id"], "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result), flush=True)
    print(json.dumps({"done": True, "program": f"pyscf {pyscf.__version__}"}), flush=True)


if __name__ == "__main__":
    main()
