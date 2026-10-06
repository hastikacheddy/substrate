"""Molecular scale: atoms on a potential-energy surface -> stationary points, normal modes, thermochemistry, TST rates.

The model is a collinear heavy-atom / proton / heavy-atom triatomic (A-H-B) moving on a tabulated 2D surface
E(x, R), with x = z_H - (z_A + z_B)/2 the proton displacement from the heavy-atom midpoint and R = z_B - z_A the
heavy-atom distance. Unlike a 1D scan, the heavy atoms are free to relax, so the barrier is the *relaxed* one.

What it does, in order:
  1. spline the table (bicubic) and trace the relaxed profile: for each x, the R that minimises E
  2. take reactant / product / transition state from that profile and refine each to a true stationary point
     (Newton on the spline gradient)
  3. harmonic vibrational analysis at each point: Hessian -> Cartesian -> mass-weighted -> remove translation
  4. vibrational free energies (zero-point energy included) and transition-state-theory rate constants

Classical TST: there is NO tunnelling correction. `tunnelling_parameter` reports how badly that matters.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import null_space

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem
from ..pes import locate_wells
from ..units import HBAR2_2AMU, H_EV_S, KB_EV, WAVENUMBER_PER_EV

KIND = "molecular.collinear_triatomic"

# internal coordinates q = (x, R) in terms of Cartesian positions (z_A, z_H, z_B):
#   x = z_H - (z_A + z_B) / 2,   R = z_B - z_A
_JACOBIAN = np.array([[-0.5, 1.0, -0.5], [-1.0, 0.0, 1.0]])


def vibrational_eigenvalues(hessian_q: np.ndarray, mass_donor: float, mass_h: float, mass_acceptor: float) -> np.ndarray:
    """Eigenvalues (eV / (angstrom^2 amu), ascending) of the two vibrational modes at a stationary point.

    `hessian_q` is the 2x2 second-derivative matrix in (x, R). It is carried to Cartesian coordinates, mass-weighted,
    and projected off the (exactly zero) translation, leaving the internal vibrations. A negative eigenvalue is an
    imaginary mode.
    """
    hessian_z = _JACOBIAN.T @ hessian_q @ _JACOBIAN
    inv_sqrt_m = 1.0 / np.sqrt([mass_donor, mass_h, mass_acceptor])
    weighted = hessian_z * np.outer(inv_sqrt_m, inv_sqrt_m)
    translation = np.sqrt([mass_donor, mass_h, mass_acceptor])[None, :]      # mass-weighted rigid shift of all atoms
    basis = null_space(translation)                                           # 3 x 2, orthonormal, internal motions
    return np.linalg.eigvalsh(basis.T @ weighted @ basis)


def signed_energy_quanta(eigenvalues: np.ndarray) -> np.ndarray:
    """hbar*omega in eV from mass-weighted Hessian eigenvalues; imaginary modes are returned negative."""
    return np.sign(eigenvalues) * np.sqrt(2.0 * HBAR2_2AMU * np.abs(eigenvalues))


def vibrational_free_energy(quanta: np.ndarray, kt: float) -> float:
    """Free energy (eV, relative to the potential minimum) of independent harmonic modes, zero-point included:
    sum of  hbar*omega/2 + kT ln(1 - exp(-hbar*omega/kT))."""
    q = np.asarray(quanta, dtype=float)
    return float(np.sum(0.5 * q + kt * np.log1p(-np.exp(-q / kt))))


class _Surface:
    """Bicubic spline of the table with analytic gradient and Hessian."""

    def __init__(self, x: np.ndarray, r: np.ndarray, energy: np.ndarray):
        self.spline = RectBivariateSpline(x, r, energy, kx=3, ky=3)
        self.x_range, self.r_range = (x[0], x[-1]), (r[0], r[-1])

    def value(self, q) -> float:
        return float(self.spline.ev(q[0], q[1]))

    def gradient(self, q) -> np.ndarray:
        return np.array([self.spline.ev(q[0], q[1], dx=1), self.spline.ev(q[0], q[1], dy=1)])

    def hessian(self, q) -> np.ndarray:
        xx, xr, rr = (self.spline.ev(q[0], q[1], dx=a, dy=b) for a, b in ((2, 0), (1, 1), (0, 2)))
        return np.array([[xx, xr], [xr, rr]])

    def inside(self, q) -> bool:
        return self.x_range[0] < q[0] < self.x_range[1] and self.r_range[0] < q[1] < self.r_range[1]

    def stationary_point(self, q0, label: str, tol: float = 1e-8, max_iter: int = 60) -> np.ndarray:
        """Newton iteration on the gradient (converges to minima and saddle points alike from a near start)."""
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            g = self.gradient(q)
            if np.linalg.norm(g) < tol:
                break
            try:
                q = q - np.linalg.solve(self.hessian(q), g)
            except np.linalg.LinAlgError:
                raise ValidationError(f"{label}: singular Hessian during the stationary-point search") from None
            if not self.inside(q):
                raise ValidationError(f"{label}: the stationary-point search left the table; widen the scan range")
        else:
            raise ValidationError(f"{label}: stationary-point search did not converge")
        return q


class StationaryPointEngine(Engine):
    name = "molecular.stationary_points_tst"
    scale = Scale.MOLECULAR
    kinds = (KIND,)
    approximations = (
        "collinear A-H-B triatomic: no rotation, bending or environment; electronic degeneracy 1",
        "the surface is a bicubic spline of the table, so scan resolution limits accuracy",
        "the transition state is the maximum of the relaxed (minimum over R) profile along x, refined to a saddle point",
        "harmonic vibrational analysis at each stationary point: anharmonicity (large for an O-H stretch) is neglected",
        "classical transition-state theory: no tunnelling and no recrossing (transmission coefficient 1)",
        "vibrations only: translational and rotational partition functions are taken to cancel between reactant and TS",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        m_d = system.param("mass_donor", "amu")
        m_h = system.param("mass_hydrogen", "amu")
        m_a = system.param("mass_acceptor", "amu")
        temperature = system.param("temperature", "K")
        x = np.asarray(system.param("surface_x", "angstrom"), dtype=float)
        r = np.asarray(system.param("surface_r", "angstrom"), dtype=float)
        energy = np.asarray(system.param("surface_energy", "eV"), dtype=float)
        if min(m_d, m_h, m_a, temperature) <= 0:
            raise ValidationError(f"{system.name}: masses and temperature must be positive")
        if energy.shape != (len(x), len(r)) or len(x) < 21 or len(r) < 11:
            raise ValidationError(
                f"{system.name}: surface_energy must be a (len(surface_x), len(surface_r)) table of at least 21 x 11 points")
        if not (np.all(np.diff(x) > 0) and np.all(np.diff(r) > 0) and np.all(np.isfinite(energy))):
            raise ValidationError(f"{system.name}: surface axes must be strictly increasing and the energies finite")

        surface = _Surface(x, r, energy)
        r_fine = np.linspace(r[0], r[-1], 901)
        fine = surface.spline(x, r_fine)                                    # (len(x), 901)
        j_best = fine.argmin(axis=1)
        mep_energy = fine[np.arange(len(x)), j_best]
        mep_r = r_fine[j_best]
        wells = locate_wells(mep_energy)
        if wells is None:
            raise ValidationError(
                f"{system.name}: the relaxed surface has fewer than two wells, so there is no reactant/product pair"
            )

        points = {}
        for label, i in zip(("reactant", "ts", "product"), wells):
            q = surface.stationary_point([x[i], mep_r[i]], f"{system.name}: {label}")
            lam = vibrational_eigenvalues(surface.hessian(q), m_d, m_h, m_a)
            n_neg = int(np.sum(lam < 0))
            if label == "ts" and n_neg != 1:
                raise ValidationError(f"{system.name}: the barrier top is not a first-order saddle point ({n_neg} imaginary modes)")
            if label != "ts" and n_neg != 0:
                raise ValidationError(f"{system.name}: the {label} geometry is not a minimum ({n_neg} imaginary modes)")
            points[label] = (q, surface.value(q), signed_energy_quanta(lam))

        kt = KB_EV * temperature
        e = {k: v[1] for k, v in points.items()}
        quanta = {k: v[2] for k, v in points.items()}
        real = {k: q[q > 0] for k, q in quanta.items()}                      # the imaginary mode is not a vibration
        zpe = {k: 0.5 * float(np.sum(real[k])) for k in points}
        g = {k: e[k] + vibrational_free_energy(real[k], kt) for k in points}
        dg_f, dg_r = g["ts"] - g["reactant"], g["ts"] - g["product"]
        prefactor = kt / H_EV_S                                              # kT/h in 1/s
        omega_ts = -float(quanta["ts"][quanta["ts"] < 0][0])                 # |hbar*omega| of the imaginary mode, eV

        src = self.name
        obs = {
            "mep_x": Quantity(x, "angstrom", source=src),
            "mep_r": Quantity(mep_r, "angstrom", source=src),
            "mep_energy": Quantity(mep_energy, "eV", source=src),
            "barrier_classical": Quantity(e["ts"] - e["reactant"], "eV", source=src),
            "barrier_zpe": Quantity(e["ts"] + zpe["ts"] - e["reactant"] - zpe["reactant"], "eV", source=src),
            "reaction_energy_classical": Quantity(e["product"] - e["reactant"], "eV", source=src),
            "reaction_energy_zpe": Quantity(e["product"] + zpe["product"] - e["reactant"] - zpe["reactant"], "eV", source=src),
            "imaginary_frequency": Quantity(omega_ts * WAVENUMBER_PER_EV, "cm^-1", source=src),
            "tunnelling_parameter": Quantity(omega_ts / kt, "1", source=src),
            "delta_g_forward": Quantity(dg_f, "eV", source=src),
            "delta_g_reverse": Quantity(dg_r, "eV", source=src),
            "rate_tst_forward": Quantity(prefactor * np.exp(-dg_f / kt), "1/s", source=src),
            "rate_tst_reverse": Quantity(prefactor * np.exp(-dg_r / kt), "1/s", source=src),
        }
        for k, (q, energy_k, _) in points.items():
            obs[f"geometry_{k}"] = Quantity(q, "angstrom", source=src)       # (x, R)
            obs[f"energy_{k}"] = Quantity(energy_k, "eV", source=src)
            obs[f"frequencies_{k}"] = Quantity(real[k] * WAVENUMBER_PER_EV, "cm^-1", source=src)
            obs[f"zpe_{k}"] = Quantity(zpe[k], "eV", source=src)
            obs[f"free_energy_{k}"] = Quantity(g[k], "eV", source=src)
        return system.evolve(
            observables=obs,
            dynamics="stationary points of E(x, R); harmonic normal modes; k = (kT/h) exp(-dG_TS / kT)",
        )
