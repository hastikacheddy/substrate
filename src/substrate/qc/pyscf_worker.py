"""Standalone PySCF worker.

Runs wherever PySCF is installed, including a Linux environment reached through WSL, so it must not import anything from
`substrate`. Protocol: a JSON object {"jobs": [...]} on stdin; one JSON line per result on stdout, then {"done": true}.

    job:    {"id", "atoms": [[symbol, x, y, z], ...] (angstrom), "charge", "spin" (2S), "theory", "basis"}
    result: {"id", "energy" (hartree), "converged", "homo_lumo_gap" (hartree), "seconds"}   or   {"id", "error"}

theory is "hf", "mp2", or "dft:<functional>" (e.g. "dft:b3lyp"); open-shell systems (spin > 0) use the unrestricted variants.
"""
from __future__ import annotations

import json
import sys
import time


def _orbital_gap(mf) -> float | None:
    """HOMO-LUMO gap (hartree) from the converged mean-field orbitals; None if there is no virtual orbital."""
    import numpy as np

    energies, occupations = np.atleast_2d(mf.mo_energy), np.atleast_2d(mf.mo_occ)
    occupied = energies[occupations > 0]
    virtual = energies[occupations == 0]
    if occupied.size == 0 or virtual.size == 0:
        return None
    return float(virtual.min() - occupied.max())


def run_job(job: dict) -> dict:
    from pyscf import dft, gto, scf

    started = time.perf_counter()
    mol = gto.M(
        atom=[[sym, (x, y, z)] for sym, x, y, z in job["atoms"]],
        charge=int(job["charge"]), spin=int(job["spin"]), basis=job["basis"], unit="Angstrom", verbose=0,
    )
    theory = job["theory"].lower()
    open_shell = mol.spin != 0
    if theory.startswith("dft:"):
        mf = (dft.UKS if open_shell else dft.RKS)(mol)
        mf.xc = theory[4:]
    elif theory in ("hf", "mp2"):
        mf = (scf.UHF if open_shell else scf.RHF)(mol)
    else:
        raise ValueError(f"unknown theory '{job['theory']}' (use 'hf', 'mp2' or 'dft:<functional>')")
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:                                  # second-order SCF is slower but far more robust
        mf = mf.newton()
        mf.kernel()
    energy = mf.e_tot
    if theory == "mp2" and mf.converged:
        from pyscf import mp
        correlated = mp.UMP2(mf) if open_shell else mp.MP2(mf)
        correlated.kernel()
        energy = correlated.e_tot
    return {
        "id": job["id"], "energy": float(energy), "converged": bool(mf.converged),
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
