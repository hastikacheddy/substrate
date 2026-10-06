"""Physical constants in the unit system the engines work in: eV, angstrom, amu, K, s.

Units are *checked*, not converted: a Quantity carries a unit string and engines
refuse inputs in the wrong unit (see ScientificSystem.param). That keeps unit errors
loud without pretending to be a units library.
"""
from scipy import constants as _c

KB_EV = _c.k / _c.e                 # Boltzmann constant, eV / K
H_EV_S = _c.h / _c.e                # Planck constant, eV s
# hbar^2 / (2 * 1 amu) in eV angstrom^2; divide by the mass in amu for hbar^2 / 2m.
HBAR2_2AMU = _c.hbar**2 / (2.0 * _c.atomic_mass) / _c.e * 1e20


def hbar2_over_2m(mass_amu: float) -> float:
    """hbar^2 / 2m in eV angstrom^2 for a mass given in amu."""
    return HBAR2_2AMU / mass_amu


WAVENUMBER_PER_EV = _c.e / (_c.h * _c.c) / 100.0   # cm^-1 per eV (1 eV = 8065.54 cm^-1)
