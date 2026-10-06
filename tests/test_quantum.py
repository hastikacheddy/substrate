import numpy as np
import pytest

from substrate import ClassicalBackend, ValidationError
from substrate.engines.quantum import DoubleWellEngine, barrier_transmission
from substrate.units import H_EV_S, hbar2_over_2m

from conftest import DEUTERON, double_well

T = hbar2_over_2m(1.007276)


def solve(**kw):
    return DoubleWellEngine().solve(double_well(**kw), ClassicalBackend())


def test_hbar2_over_2m_for_proton():
    assert hbar2_over_2m(1.007276) == pytest.approx(0.002075, rel=1e-3)


def test_harmonic_oscillator_levels():
    k = 20.0
    x = np.linspace(-1.0, 1.0, 1500)
    hop = T / (x[1] - x[0]) ** 2
    e, _ = ClassicalBackend().lowest_eigenpairs(0.5 * k * x**2 + 2 * hop, np.full(len(x) - 1, -hop), 5)
    hw = np.sqrt(2 * T * k)
    assert e == pytest.approx(hw * (np.arange(5) + 0.5), rel=1e-3)


@pytest.mark.parametrize("energy", [0.1, 0.3, 0.49, 0.6, 1.0])
def test_transmission_matches_rectangular_barrier(energy):
    v0, width, cells = 0.5, 0.5, 400
    got = barrier_transmission(np.full(cells, v0), width / cells, [energy], 0.0, 0.0, T)[0]
    if energy < v0:
        kappa = np.sqrt((v0 - energy) / T)
        exact = 1 / (1 + v0**2 * np.sinh(kappa * width) ** 2 / (4 * energy * (v0 - energy)))
    else:
        k2 = np.sqrt((energy - v0) / T)
        exact = 1 / (1 + v0**2 * np.sin(k2 * width) ** 2 / (4 * energy * (energy - v0)))
    assert got == pytest.approx(exact, rel=1e-9)


def test_transmission_stays_accurate_for_very_thick_barrier():
    # 1 - |R|^2 would round this to exactly 0; the amplitude recursion must not
    width, cells = 8.0, 400
    got = barrier_transmission(np.full(cells, 2.0), width / cells, [0.1], 0.0, 0.0, T)[0]
    kappa = np.sqrt(1.9 / T)
    exact = 1 / (1 + 4.0 * np.sinh(kappa * width) ** 2 / (4 * 0.1 * 1.9))
    assert 0 < got < 1e-100
    assert got == pytest.approx(exact, rel=1e-6)


def test_transmission_is_zero_at_or_below_an_asymptote():
    got = barrier_transmission(np.full(50, 0.5), 0.01, [0.05, 0.15, 0.3], 0.0, 0.1, T)
    assert got[0] == 0.0 and got[1] > 0.0 and got[2] > 0.0


def test_symmetric_well_levels_pair_with_definite_parity():
    out = solve(d_e=0.0, barrier=0.5)
    e = out.obs("energy_levels")
    psi = out.state["wavefunctions"].value
    assert 0 < e[1] - e[0] < 0.1 * (e[2] - e[1])           # tight doublet, then a gap
    assert np.allclose(psi[0], psi[0][::-1], atol=1e-8)    # ground state even
    assert np.allclose(psi[1], -psi[1][::-1], atol=1e-8)   # first excited odd


def test_wavefunctions_are_normalised():
    out = solve()
    x = out.state["x_grid"].value
    psi = out.state["wavefunctions"].value
    assert (psi**2).sum(axis=1) * (x[1] - x[0]) == pytest.approx(np.ones(len(psi)), abs=1e-9)


def test_splitting_shrinks_as_barrier_grows():
    splits = [solve(d_e=0.0, barrier=v).obs("splitting_01") for v in (0.3, 0.5, 0.8)]
    assert splits[0] > splits[1] > splits[2] > 0


@pytest.mark.parametrize("barrier", [0.4, 0.6, 0.8])
def test_splitting_from_diagonalisation_agrees_with_transmission_via_wkb(barrier):
    # Two independent routes to the same barrier: eigenvalues of H (splitting) and the scattering
    # recursion (transmission). WKB links them: splitting ~ (hbar*omega/pi) * sqrt(P(E0)).
    out = solve(d_e=0.0, barrier=barrier)
    hbar_omega = out.obs("attempt_frequency_left")[0] * H_EV_S
    wkb = hbar_omega / np.pi * np.sqrt(out.obs("transmission_left")[0])
    assert out.obs("splitting_01") == pytest.approx(wkb, rel=0.15)


def test_deuteron_tunnels_less_than_proton():
    p_h = solve().obs("transmission_left")[0]
    p_d = solve(mass=DEUTERON).obs("transmission_left")[0]
    assert 0 < p_d < p_h * 1e-2


def test_localised_levels_sit_just_above_the_full_system_levels():
    out = solve()
    # Dirichlet truncation at the barrier top can only push levels up, never below the true spectrum
    assert out.obs("levels_left")[0] >= out.obs("ground_state_energy") - 1e-12
    assert out.obs("zpe_left") == pytest.approx(out.obs("levels_left")[0] - out.obs("well_min_left"))


def test_engine_rejects_wrong_unit_and_bad_values():
    bad_unit = double_well()
    bad_unit.parameters["barrier_height"].unit = "kcal/mol"
    with pytest.raises(ValidationError, match="expected 'eV'"):
        DoubleWellEngine().solve(bad_unit, ClassicalBackend())
    with pytest.raises(ValidationError, match="positive"):
        solve(barrier=-0.1)
    with pytest.raises(ValidationError, match="n_grid"):
        solve(n_grid=10)
