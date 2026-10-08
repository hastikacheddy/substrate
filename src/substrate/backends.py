"""Solver backends: the seam where classical, GPU, HPC, or quantum hardware plug in.

Engines never call scipy directly for the heavy numerics; they ask a SolverBackend.
Only a classical CPU backend ships today. A Qiskit backend would implement
`lowest_eigenpairs` (e.g. VQE/QPE on a discretised Hamiltonian) behind the same signature.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal


class SolverBackend(ABC):
    name: str

    @abstractmethod
    def lowest_eigenpairs(self, diag: np.ndarray, offdiag: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k lowest eigenvalues (ascending) and eigenvectors (columns) of a symmetric tridiagonal matrix."""

    @abstractmethod
    def symmetric_eigh(self, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All eigenpairs of a batch of small real symmetric matrices, shape (..., n, n).
        Eigenvalues ascending along the last axis; eigenvectors are the columns of each result matrix.
        This is the electronic-structure seam: the place a VQE/QPE solver would plug in."""

    @abstractmethod
    def integrate_ode(self, rhs: Callable, y0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """Integrate dy/dt = rhs(t, y) from t_eval[0]; returns y sampled at t_eval, shape (len(t_eval), len(y0))."""


#: SciPy's LSODA wraps a Fortran routine with global state: one integration per process at a time. Without this lock two threads (two browser
#: tabs on the GUI server, say) fail with "Integrator `lsoda` can be used to solve only one problem at a time".
_ODE_LOCK = threading.Lock()


class ClassicalBackend(SolverBackend):
    name = "classical"

    def lowest_eigenpairs(self, diag, offdiag, k):
        k = min(k, len(diag))
        return eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, k - 1))

    def symmetric_eigh(self, matrices):
        return np.linalg.eigh(matrices)

    def integrate_ode(self, rhs, y0, t_eval):
        with _ODE_LOCK:
            sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, method="LSODA", t_eval=t_eval, rtol=1e-9, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol.y.T


_BACKENDS: dict[str, type[SolverBackend]] = {"classical": ClassicalBackend}


def register_backend(cls: type[SolverBackend]) -> type[SolverBackend]:
    _BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str = "classical") -> SolverBackend:
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise KeyError(f"no backend '{name}' (registered: {', '.join(_BACKENDS)})") from None
