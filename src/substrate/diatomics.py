"""Free X-H diatomics: the Morse curve a bond would have with no hydrogen-bond partner, from spectroscopy.

The two-state model gives each diabatic state a Morse curve. Calibration fits its depth, width and equilibrium length to a surface; this
module supplies the physical alternative: the same three numbers for the free diatomic OH, NH, HF or HCl, so a fit can be anchored to them
(`anchored_bounds`) and the cost of anchoring measured. A hydrogen-bonded X-H bond is longer and softer than the free one, so these are a
reference point and not the right answer for a complex.

Inputs. ``omega_e``, ``omega_e_x_e`` and ``r_e`` are the ground-state constants of the NIST Chemistry WebBook (Huber and Herzberg; checked
against the WebBook pages for H19F, H35Cl, 16OH and 14NH). ``d0`` is the dissociation energy from the ground vibrational level, in eV: HF from
the WebBook's limiting-curve value (47333 +- 60 cm-1); OH, HCl and NH from standard heats of formation (about +-0.03 eV; for NH the WebBook's
own note spans 3.17-3.59 eV, so +-0.1 is the honest uncertainty). The Morse well depth is D_e = D_0 + the zero-point energy,
omega_e/2 - omega_e x_e/4, and the width follows from the harmonic frequency: omega = alpha sqrt(2 D_e / mu), the Morse curvature 2 D_e alpha^2 = mu omega^2.
The width scales as D_e^(-1/2), so a 0.1 eV uncertainty in D_e is 1.4-1.7% in alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ValidationError

_C_CM_PER_S = 2.99792458e10          # speed of light, cm/s
_AMU_KG = 1.66053906660e-27
_EV_J = 1.602176634e-19
_EV_PER_WAVENUMBER = 1.239841984e-4  # eV per cm-1
_HYDROGEN_MASS = 1.00782503          # amu

_HEAVY_MASS = {"O": 15.9949146, "N": 14.0030740, "F": 18.9984032, "Cl": 34.9688527}     # amu, most abundant isotope


@dataclass(frozen=True)
class Diatomic:
    heavy: str                    # the element bonded to the hydrogen
    omega_e: float                # cm-1
    omega_e_x_e: float            # cm-1
    r_e: float                    # angstrom
    d0: float                     # eV, from the ground vibrational level
    d0_uncertainty: float         # eV

    @property
    def reduced_mass(self) -> float:
        """amu."""
        m = _HEAVY_MASS[self.heavy]
        return m * _HYDROGEN_MASS / (m + _HYDROGEN_MASS)

    @property
    def zero_point(self) -> float:
        """eV: omega_e/2 - omega_e x_e/4, the zero-point energy of the anharmonic oscillator."""
        return (0.5 * self.omega_e - 0.25 * self.omega_e_x_e) * _EV_PER_WAVENUMBER

    @property
    def depth(self) -> float:
        """The Morse well depth D_e, eV."""
        return self.d0 + self.zero_point

    @property
    def alpha(self) -> float:
        """The Morse width, 1/angstrom: omega = alpha sqrt(2 D_e / mu) with omega = 2 pi c omega_e."""
        omega = 2.0 * math.pi * _C_CM_PER_S * self.omega_e                                       # rad/s
        return omega * math.sqrt(self.reduced_mass * _AMU_KG / (2.0 * self.depth * _EV_J)) * 1e-10


DIATOMICS: dict[str, Diatomic] = {
    "O": Diatomic("O", omega_e=3737.761, omega_e_x_e=84.8813, r_e=0.96966, d0=4.40, d0_uncertainty=0.03),
    "N": Diatomic("N", omega_e=3282.27, omega_e_x_e=78.35, r_e=1.03621, d0=3.42, d0_uncertainty=0.10),
    "F": Diatomic("F", omega_e=4138.32, omega_e_x_e=89.88, r_e=0.916808, d0=5.869, d0_uncertainty=0.01),
    "Cl": Diatomic("Cl", omega_e=2990.9463, omega_e_x_e=52.8186, r_e=1.274552, d0=4.434, d0_uncertainty=0.03),
}


def morse_parameters(heavy: str) -> dict[str, float]:
    """The free X-H diatomic's Morse curve in the model's names: morse_depth (eV), morse_alpha (1/angstrom), morse_r_eq (angstrom)."""
    if heavy not in DIATOMICS:
        raise ValidationError(f"no free-diatomic parameters for {heavy}-H (have: {', '.join(DIATOMICS)})")
    d = DIATOMICS[heavy]
    return {"morse_depth": d.depth, "morse_alpha": d.alpha, "morse_r_eq": d.r_e}


def anchored_bounds(molecule: dict, *, depth: bool = False, tolerance: float = 1e-9) -> tuple[tuple[str, float, float], ...]:
    """`FitSettings.bounds` that hold a fit's bonds at the free diatomics' values: the donor atom's X-H for ``morse_*`` and the acceptor atom's
    for ``acceptor_morse_*``. The equilibrium length and width are held; the depth too if ``depth`` (otherwise it stays free, because a window
    of a few eV above the minimum barely sees the wall). A donor and acceptor of one element give the same values, so the shared-bond model
    suffices, and only the donor's bounds are returned. Use ``bonds="separate"`` when they differ."""
    atoms = molecule["atoms"]
    donor, acceptor = atoms[molecule["donor"]][0], atoms[molecule["acceptor"]][0]
    names = ("morse_r_eq", "morse_alpha") + (("morse_depth",) if depth else ())
    out = []
    for prefix, heavy in (("", donor), ("acceptor_", acceptor)):
        if prefix and heavy == donor:
            continue
        values = morse_parameters(heavy)
        out += [(prefix + name, values[name] - tolerance, values[name] + tolerance) for name in names]
    return tuple(out)


def needs_separate_bonds(molecule: dict) -> bool:
    """Whether the donor and acceptor atoms differ, so one Morse curve cannot describe both bonds."""
    atoms = molecule["atoms"]
    return atoms[molecule["donor"]][0] != atoms[molecule["acceptor"]][0]
