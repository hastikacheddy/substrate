"""Quantum scale: nuclear quantum dynamics of one particle (a proton) on a 1D double-well PES.

Solves  H = -(hbar^2/2m) d^2/dx^2 + V(x)  by finite differences and extracts what the next scale
needs: level structure per well, and the barrier transmission probability at those levels.

Two engines share one solver core and differ only in where V(x) comes from:
  quantum.double_well_1d  analytic quartic  V = V0 (1 - (x/a)^2)^2 + (dE/2)(x/a)
  quantum.tabulated_1d    a tabulated surface, e.g. computed by the electronic-structure scale

Left well (lower x) is the reactant, right well the product.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem
from ..pes import MIN_PROMINENCE_EV, locate_wells, prominent_minima
from ..units import H_EV_S, hbar2_over_2m

KIND = "quantum.double_well_1d"
KIND_TABULATED = "quantum.tabulated_1d"


def double_well(x: np.ndarray, barrier: float, half_sep: float, reaction_energy: float = 0.0) -> np.ndarray:
    u = x / half_sep
    return barrier * (1.0 - u**2) ** 2 + 0.5 * reaction_energy * u


def barrier_transmission(
    v_cells: np.ndarray, dx: float, energies: np.ndarray, v_left: float, v_right: float, hbar2_2m: float
) -> np.ndarray:
    """Transmission probability through a piecewise-constant 1D potential with flat asymptotes.

    Cells of width dx at potentials `v_cells`, flanked by semi-infinite media at v_left / v_right.
    Uses the Parratt reflection recursion with an explicit amplitude product, which stays stable and
    accurate for thick barriers (T ~ 1e-30) where 1 - |R|^2 would round to zero.
    Returns 0 for energies at or below either asymptote.
    """
    E = np.asarray(energies, dtype=float)[:, None]
    v = np.concatenate([[v_left], v_cells, [v_right]])[None, :]
    k = np.sqrt((E - v) / hbar2_2m + 0j)               # Im k >= 0: evanescent waves decay to the right
    k = np.where(k == 0, 1e-12, k)
    rho = (k[:, :-1] - k[:, 1:]) / (k[:, :-1] + k[:, 1:])   # Fresnel coefficient at each interface
    n_cells = len(v_cells)

    gamma = np.zeros(E.shape[0], dtype=complex)              # reflection seen from the left of interface j
    log_amp = np.zeros(E.shape[0])                           # log |transmission amplitude|
    for j in range(n_cells, -1, -1):
        if j == n_cells:
            back = np.zeros_like(gamma)                      # right medium is semi-infinite: nothing reflects back
        else:
            back = gamma * np.exp(2j * k[:, j + 1] * dx)
            log_amp -= k[:, j + 1].imag * dx                 # decay across layer j+1
        denom = 1.0 + rho[:, j] * back
        log_amp += np.log(np.abs(1.0 + rho[:, j])) - np.log(np.abs(denom))
        gamma = (rho[:, j] + back) / denom

    flux_ratio = k[:, -1].real / k[:, 0].real
    T = flux_ratio * np.exp(2.0 * log_amp)
    T = np.where((E[:, 0] <= v_left) | (E[:, 0] <= v_right), 0.0, T)
    return np.clip(T, 0.0, 1.0)


def attempt_frequencies(levels: np.ndarray, curvature: float, hbar2_2m: float) -> np.ndarray:
    """Classical oscillation frequency in the well at each level, nu_n = (dE/dn)/h (Hz).

    Level-spacing derivative from the computed levels; harmonic estimate from the well
    curvature (eV/angstrom^2) when only one level exists.
    """
    n = len(levels)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([np.sqrt(2.0 * hbar2_2m * curvature) / H_EV_S])
    dEdn = np.gradient(levels, edge_order=2 if n >= 3 else 1)
    return dEdn / H_EV_S


class _Schrodinger1D(Engine):
    """Shared solver core: potential V(x) on a uniform grid in, levels / wells / transmission out."""

    scale = Scale.QUANTUM
    approximations = (
        "one nuclear coordinate; every other degree of freedom is frozen",
        "Born-Oppenheimer: electrons enter only through the potential V(x)",
        "time-independent Schrodinger equation by second-order finite differences",
        "wells are the two lowest minima with at least 5 meV of prominence; a third well is not modelled",
        "well-localised levels from half-domains truncated (Dirichlet) at the barrier top",
        "barrier transmission by scattering between the two minima with flat asymptotes at the minima",
    )

    def _solve_surface(
        self,
        system: ScientificSystem,
        x: np.ndarray,
        V: np.ndarray,
        mass: float,
        n_levels: int,
        min_prominence: float,
        backend: SolverBackend,
    ) -> ScientificSystem:
        n_grid = len(x)
        t = hbar2_over_2m(mass)
        dx = x[1] - x[0]
        hop = t / dx**2

        def lowest(v_slice, k):
            return backend.lowest_eigenpairs(v_slice + 2.0 * hop, np.full(len(v_slice) - 1, -hop), k)

        wells = locate_wells(V, min_prominence)
        if wells is None:
            raise ValidationError(
                f"{system.name}: the potential has fewer than two wells (>= {min_prominence * 1e3:g} meV "
                f"prominent), so there is no reactant/product pair"
            )
        i_left, i_top, i_right = wells
        v_top = float(V[i_top])

        evals, evecs = lowest(V, n_levels)
        psi = evecs.T / np.sqrt(dx)                         # normalised: sum psi^2 dx = 1
        left, _ = lowest(V[:i_top], n_levels)
        right, _ = lowest(V[i_top + 1:], n_levels)
        left, right = left[left < v_top], right[right < v_top]

        p_left = barrier_transmission(V[i_left:i_right + 1], dx, left, V[i_left], V[i_right], t)
        curvature = (V[i_left - 1] - 2.0 * V[i_left] + V[i_left + 1]) / dx**2
        nu_left = attempt_frequencies(left, curvature, t)

        src = self.name
        obs = {
            "energy_levels": Quantity(evals, "eV", source=src),
            "ground_state_energy": Quantity(evals[0], "eV", source=src),
            "splitting_01": Quantity(evals[1] - evals[0], "eV", source=src),
            "barrier_top": Quantity(v_top, "eV", source=src),
            "well_min_left": Quantity(V[i_left], "eV", source=src),
            "well_min_right": Quantity(V[i_right], "eV", source=src),
            "n_wells": Quantity(len(prominent_minima(V, min_prominence)), "count", source=src),
            "levels_left": Quantity(left, "eV", source=src),
            "levels_right": Quantity(right, "eV", source=src),
            "transmission_left": Quantity(p_left, "1", source=src),
            "attempt_frequency_left": Quantity(nu_left, "Hz", source=src),
        }
        if len(left):
            obs["zpe_left"] = Quantity(left[0] - V[i_left], "eV", source=src)
        if len(left) and len(right):
            obs["reaction_energy_zpe"] = Quantity(right[0] - left[0], "eV", source=src)

        state = {
            "x_grid": Quantity(x, "angstrom", source=src),
            "potential": Quantity(V, "eV", source=src),
            "wavefunctions": Quantity(psi, "angstrom^-1/2", source=src),
        }
        return system.evolve(
            observables=obs,
            state=state,
            dynamics="(-hbar^2/2m d^2/dx^2 + V(x)) psi = E psi",
        )

    @staticmethod
    def _grid_controls(system: ScientificSystem) -> tuple[int, int, float]:
        n_grid = int(system.param("n_grid", default=1500))
        n_levels = int(system.param("n_levels", default=8))
        min_prominence = float(system.param("min_prominence", "eV", default=MIN_PROMINENCE_EV))
        if n_grid < 200 or n_levels < 2:
            raise ValidationError(f"{system.name}: need n_grid >= 200 and n_levels >= 2")
        return n_grid, n_levels, min_prominence


class DoubleWellEngine(_Schrodinger1D):
    name = KIND
    kinds = (KIND,)
    approximations = _Schrodinger1D.approximations + (
        "the potential is a model quartic supplied as parameters, not computed from electrons",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        mass = system.param("mass", "amu")
        v0 = system.param("barrier_height", "eV")
        a = system.param("half_separation", "angstrom")
        d_e = system.param("reaction_energy", "eV", default=0.0)
        n_grid, n_levels, min_prominence = self._grid_controls(system)
        if mass <= 0 or v0 <= 0 or a <= 0:
            raise ValidationError(f"{system.name}: mass, barrier_height and half_separation must be positive")
        x = np.linspace(-2.0 * a, 2.0 * a, n_grid)
        return self._solve_surface(system, x, double_well(x, v0, a, d_e), mass, n_levels, min_prominence, backend)


class TabulatedPotentialEngine(_Schrodinger1D):
    """Solves the nuclear problem on a tabulated surface (parameters `pes_x`, `pes_energy`)."""

    name = KIND_TABULATED
    kinds = (KIND_TABULATED,)
    approximations = _Schrodinger1D.approximations + (
        "the tabulated surface is resampled onto a uniform grid by a natural cubic spline",
        "the surface is taken to end where the table ends: it must rise well above the barrier on both sides",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        mass = system.param("mass", "amu")
        pes_x = np.asarray(system.param("pes_x", "angstrom"), dtype=float)
        pes_v = np.asarray(system.param("pes_energy", "eV"), dtype=float)
        n_grid, n_levels, min_prominence = self._grid_controls(system)
        if mass <= 0:
            raise ValidationError(f"{system.name}: mass must be positive")
        if pes_x.ndim != 1 or pes_x.shape != pes_v.shape or len(pes_x) < 8:
            raise ValidationError(f"{system.name}: pes_x and pes_energy must be equal-length 1D tables (>= 8 points)")
        if not np.all(np.diff(pes_x) > 0) or not np.all(np.isfinite(pes_v)):
            raise ValidationError(f"{system.name}: pes_x must be strictly increasing and pes_energy finite")
        x = np.linspace(pes_x[0], pes_x[-1], n_grid)
        return self._solve_surface(system, x, CubicSpline(pes_x, pes_v)(x), mass, n_levels, min_prominence, backend)
