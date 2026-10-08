"""Free X-H diatomic Morse parameters: checked against physics that does not use the module's own formula."""
import math

import numpy as np
import pytest

from substrate import ValidationError
from substrate.diatomics import DIATOMICS, anchored_bounds, morse_parameters, needs_separate_bonds
from substrate.engines.electronic import morse
from substrate.molecules import TEMPLATES

CM_PER_S, AMU, EV = 2.99792458e10, 1.66053906660e-27, 1.602176634e-19


@pytest.mark.parametrize("heavy", sorted(DIATOMICS))
def test_the_morse_curve_reproduces_the_harmonic_frequency_when_its_curvature_is_measured_numerically(heavy):
    """The curvature at the minimum, found by finite differences of the same `morse` the model uses, must give back the measured omega_e through
    omega = sqrt(k / mu): an independent route to the width, not the module's formula read back."""
    d, p = DIATOMICS[heavy], morse_parameters(heavy)
    h = 1e-4                                                                                       # angstrom
    curvature_ev_per_a2 = (morse(p["morse_r_eq"] + h, **_args(p)) - 2 * morse(p["morse_r_eq"], **_args(p)) + morse(p["morse_r_eq"] - h, **_args(p))) / h**2
    k = curvature_ev_per_a2 * EV / 1e-20                                                           # J / m^2
    wavenumber = math.sqrt(k / (d.reduced_mass * AMU)) / (2 * math.pi * CM_PER_S)
    assert wavenumber == pytest.approx(d.omega_e, rel=2e-4)


def _args(p):
    return {"depth": p["morse_depth"], "alpha": p["morse_alpha"], "r_eq": p["morse_r_eq"]}


@pytest.mark.parametrize("heavy", sorted(DIATOMICS))
def test_the_well_depth_and_width_are_consistent_with_the_measured_anharmonicity(heavy):
    """A Morse oscillator has omega_e x_e = omega_e^2 / (4 D_e): an independent check on D_e, which a wrong dissociation energy would fail.
    Real bonds are not exactly Morse, so 20% is allowed (they come out 3-18% apart)."""
    d = DIATOMICS[heavy]
    morse_anharmonicity = d.omega_e**2 / (4.0 * d.depth / 1.239841984e-4)                         # cm-1, with D_e in cm-1
    assert morse_anharmonicity == pytest.approx(d.omega_e_x_e, rel=0.20)


def test_the_four_bonds_have_the_physical_ordering():
    p = {k: morse_parameters(k) for k in DIATOMICS}
    assert p["Cl"]["morse_r_eq"] > p["N"]["morse_r_eq"] > p["O"]["morse_r_eq"] > p["F"]["morse_r_eq"]        # 1.27, 1.04, 0.97, 0.92 angstrom
    assert p["F"]["morse_depth"] > p["O"]["morse_depth"] > p["N"]["morse_depth"]                             # the strongest bond is HF
    assert p["Cl"]["morse_alpha"] < min(p[k]["morse_alpha"] for k in ("O", "N", "F"))                        # the long Cl-H bond is the softest
    assert all(1.8 < v["morse_alpha"] < 2.4 and 3.0 < v["morse_depth"] < 6.5 for v in p.values())
    assert DIATOMICS["F"].d0 == pytest.approx(47333 * 1.239841984e-4, abs=2e-3)                              # the WebBook's HF limit, 47333 cm-1


def test_unknown_elements_are_refused():
    with pytest.raises(ValidationError, match="no free-diatomic parameters for S-H"):
        morse_parameters("S")


def test_bonds_are_anchored_by_the_elements_of_the_donor_and_acceptor_atoms():
    cl = anchored_bounds(TEMPLATES["chloride_hf_anion"]())
    assert [name for name, *_ in cl] == ["morse_r_eq", "morse_alpha", "acceptor_morse_r_eq", "acceptor_morse_alpha"]
    values = {name: (lo + hi) / 2 for name, lo, hi in cl}
    assert values["morse_r_eq"] == pytest.approx(1.274552) and values["acceptor_morse_r_eq"] == pytest.approx(0.916808)
    assert all(hi - lo < 1e-8 for _, lo, hi in cl)                                                           # held, not merely bounded
    full = anchored_bounds(TEMPLATES["chloride_hf_anion"](), depth=True)
    assert [name for name, *_ in full] == ["morse_r_eq", "morse_alpha", "morse_depth", "acceptor_morse_r_eq", "acceptor_morse_alpha", "acceptor_morse_depth"]
    zundel = anchored_bounds(TEMPLATES["zundel_cation"]())                                                   # two oxygens: one curve is enough
    assert [name for name, *_ in zundel] == ["morse_r_eq", "morse_alpha"]


def test_which_templates_need_separate_bonds_follows_their_atoms():
    needs = {name: needs_separate_bonds(make()) for name, make in TEMPLATES.items()}
    assert needs == {"zundel_cation": False, "ammonium_dimer_cation": False, "bifluoride_anion": False, "water_ammonia_cation": True,
                     "methanol_water_cation": False, "ammonia_methylamine_cation": False, "fluoride_methanol_anion": True,
                     "chloride_hf_anion": True}


@pytest.mark.parametrize("heavy", sorted(DIATOMICS))
def test_the_zero_point_energy_is_the_ground_state_of_the_morse_oscillator_solved_numerically(heavy):
    """Solve -(hbar^2 / 2 mu) psi'' + V psi = E psi for the module's own Morse curve on a grid, with no reference to the zero-point formula: the lowest
    level above the well bottom must be the zero-point energy the module adds to D_0 to get D_e. A Morse oscillator is slightly more anharmonic
    than the real bond, so the two agree to a fraction of a millielectronvolt, not exactly; 3 meV is far below the error of a wrong formula."""
    from scipy.linalg import eigh_tridiagonal
    d, p = DIATOMICS[heavy], morse_parameters(heavy)
    hbar2_over_2mu = (1.054571817e-34) ** 2 / (2 * d.reduced_mass * AMU) / EV * 1e20           # eV angstrom^2
    r = np.linspace(0.45, 5.0, 3000)
    step = r[1] - r[0]
    potential = morse(r, **_args(p))
    diagonal = 2 * hbar2_over_2mu / step**2 + potential
    off = -hbar2_over_2mu / step**2 * np.ones(len(r) - 1)
    ground = eigh_tridiagonal(diagonal, off, select="i", select_range=(0, 0))[0][0]
    assert ground == pytest.approx(d.zero_point, abs=0.003)
