"""Electronic-structure scale: the Born-Oppenheimer surface for a proton moving between two heavy atoms.

Two-state empirical valence bond (EVB) model. For a proton at displacement x from the midpoint of two heavy
atoms a distance R apart, the electronic Hamiltonian is

    H = [[ V_A,    Delta ],        V_A = Morse(r_A),  V_B = Morse(r_B) + offset,
         [ Delta,  V_B   ]]        r_A = R/2 + x,     r_B = R/2 - x

where A is the donor-bound ("reactant") diabatic state and B the acceptor-bound ("product") state. Diagonalising H
gives the adiabatic ground state E0: the surface the nuclei move on. The eigenvectors give the electronic character.

Two engines share that diagonalisation:
  electronic.evb_two_state     the heavy-atom distance R is fixed: a 1D scan along x
  electronic.evb_two_state_2d  R is a coordinate too: a 2D surface E(x, R), with a distance-dependent coupling
                               Delta(R) = Delta0 exp(-beta (R - R0)) and a Morse heavy-atom interaction V_OO(R)

THESE ARE MODEL HAMILTONIANS, NOT AB INITIO CALCULATIONS. Their parameters are inputs. What they compute, by
genuine diagonalisation, is how the surface (barrier, asymmetry, well count) emerges from those parameters.
"""
from __future__ import annotations

import numpy as np

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem
from ..pes import MIN_PROMINENCE_EV, locate_wells, prominent_minima

KIND = "electronic.evb_two_state"
KIND_2D = "electronic.evb_two_state_2d"


def morse(r: np.ndarray, depth: float, alpha: float, r_eq: float) -> np.ndarray:
    return depth * (1.0 - np.exp(-alpha * (r - r_eq))) ** 2


def evb_adiabats(backend: SolverBackend, r_a, r_b, coupling, offset: float, depth: float, alpha: float, r_eq: float):
    """Diagonalise the 2x2 valence-bond Hamiltonian over any broadcastable set of geometries.

    Returns (E0, E1, |c_A|^2 of the ground state, V_A, V_B), each with the broadcast shape.
    """
    v_a = morse(np.asarray(r_a), depth, alpha, r_eq)
    v_b = morse(np.asarray(r_b), depth, alpha, r_eq) + offset
    v_a, v_b, coupling = np.broadcast_arrays(v_a, v_b, coupling)
    h = np.zeros(v_a.shape + (2, 2))
    h[..., 0, 0], h[..., 1, 1], h[..., 0, 1], h[..., 1, 0] = v_a, v_b, coupling, coupling
    evals, evecs = backend.symmetric_eigh(h)
    return evals[..., 0], evals[..., 1], evecs[..., 0, 0] ** 2, v_a, v_b


def _slice_observables(src: str, x, e0, e1, weight, v_a, v_b) -> dict[str, Quantity]:
    """Observables of a 1D scan along the proton coordinate; both engines report them identically."""
    obs = {
        "scan_coordinate": Quantity(x, "angstrom", source=src),
        "scan_energy": Quantity(e0, "eV", source=src),
        "excited_state_energy": Quantity(e1, "eV", source=src),
        "electronic_gap": Quantity(e1 - e0, "eV", source=src),
        "electronic_gap_min": Quantity((e1 - e0).min(), "eV", source=src),
        "reactant_weight": Quantity(weight, "1", source=src),               # |c_A|^2 of the ground state
        "diabatic_a": Quantity(v_a, "eV", source=src),
        "diabatic_b": Quantity(v_b, "eV", source=src),
        "n_minima": Quantity(len(prominent_minima(e0, MIN_PROMINENCE_EV)), "count", source=src),
    }
    wells = locate_wells(e0)
    if wells is not None:
        i_left, i_top, i_right = wells
        obs["classical_barrier"] = Quantity(e0[i_top] - e0[i_left], "eV", source=src)
        obs["classical_reaction_energy"] = Quantity(e0[i_right] - e0[i_left], "eV", source=src)
    return obs


class EVBProtonTransferEngine(Engine):
    name = KIND
    scale = Scale.ELECTRONIC_STRUCTURE
    kinds = (KIND,)
    approximations = (
        "model Hamiltonian, not ab initio: two valence-bond states (proton on donor, proton on acceptor) only",
        "both diabatic states are Morse O-H potentials with shared parameters, apart from a constant offset",
        "electronic coupling is constant along the scan (donor-acceptor distance is held fixed)",
        "rigid scan along the proton coordinate: the heavy atoms and any environment do not relax",
        "electronic ground state only; excited-state character is reported through the gap, not propagated",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        big_r = system.param("donor_acceptor_distance", "angstrom")
        depth = system.param("morse_depth", "eV")
        alpha = system.param("morse_alpha", "1/angstrom")
        r_eq = system.param("morse_r_eq", "angstrom")
        coupling = system.param("coupling", "eV")
        offset = system.param("diabatic_offset", "eV", default=0.0)
        r_min = system.param("scan_min_bond_length", "angstrom", default=0.6)
        n_scan = int(system.param("n_scan", default=241))

        if min(big_r, depth, alpha, r_eq, r_min) <= 0:
            raise ValidationError(f"{system.name}: distances, Morse depth and Morse alpha must be positive")
        if coupling < 0:
            raise ValidationError(f"{system.name}: coupling must be non-negative (only |coupling| affects the surface)")
        x_max = big_r / 2.0 - r_min
        if x_max <= 0:
            raise ValidationError(f"{system.name}: donor_acceptor_distance must exceed 2 * scan_min_bond_length")
        if n_scan < 50:
            raise ValidationError(f"{system.name}: need n_scan >= 50")

        x = np.linspace(-x_max, x_max, n_scan)
        e0, e1, weight, v_a, v_b = evb_adiabats(
            backend, big_r / 2.0 + x, big_r / 2.0 - x, coupling, offset, depth, alpha, r_eq
        )
        return system.evolve(
            observables=_slice_observables(self.name, x, e0, e1, weight, v_a, v_b),
            dynamics="H_el(x) c = E(x) c  (2x2 valence-bond Hamiltonian at each proton position)",
        )


class EVBFlexibleProtonTransferEngine(Engine):
    """The same valence-bond model with the donor-acceptor distance R as a second coordinate.

    Surface:  E(x, R) = E0_el(x, R; Delta(R)) + V_OO(R),   Delta(R) = Delta0 exp(-beta (R - R0)),
              V_OO a Morse potential in R: attractive at long range, steeply repulsive at short range.
    (A soft harmonic V_OO is not enough: the coupling grows exponentially as R shrinks and the surface collapses
    into a single symmetric short-distance well.)  Also reports a rigid 1D slice (`scan_*` observables) at R = `scan_distance` (default R0), so a
    frozen-heavy-atom route stays available. The slice is cut from the same surface: moving `scan_distance` never
    changes the surface (`reference_distance` defines the coupling and nothing else).
    """

    name = KIND_2D
    scale = Scale.ELECTRONIC_STRUCTURE
    kinds = (KIND_2D,)
    approximations = (
        "model Hamiltonian, not ab initio: two valence-bond states (proton on donor, proton on acceptor) only",
        "both diabatic states are Morse O-H potentials with shared parameters, apart from a constant offset",
        "collinear heavy atom - proton - heavy atom geometry; the surface is a function of (x, R) only",
        "coupling decays exponentially with R; the heavy-atom interaction is a Morse potential in R",
        "no environment or solvent; electronic ground state only (excited states enter through the gap)",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        depth = system.param("morse_depth", "eV")
        alpha = system.param("morse_alpha", "1/angstrom")
        r_eq = system.param("morse_r_eq", "angstrom")
        offset = system.param("diabatic_offset", "eV", default=0.0)
        coupling = system.param("coupling", "eV")
        decay = system.param("coupling_decay", "1/angstrom")
        r_ref = system.param("reference_distance", "angstrom")
        r_scan = system.param("scan_distance", "angstrom", default=r_ref)
        oo_depth = system.param("oo_depth", "eV")
        oo_alpha = system.param("oo_alpha", "1/angstrom")
        oo_eq = system.param("oo_equilibrium", "angstrom")
        x_ext = system.param("x_extent", "angstrom", default=0.65)
        d_min = system.param("distance_min", "angstrom", default=2.2)
        d_max = system.param("distance_max", "angstrom", default=3.1)
        n_x = int(system.param("n_x", default=161))
        n_r = int(system.param("n_r", default=91))

        if min(depth, alpha, r_eq, r_ref, oo_depth, oo_alpha, oo_eq, x_ext, d_min) <= 0:
            raise ValidationError(f"{system.name}: distances and Morse parameters must be positive")
        if coupling < 0 or decay < 0:
            raise ValidationError(f"{system.name}: coupling and coupling_decay must be non-negative")
        if not d_min < d_max or not (d_min <= r_ref <= d_max and d_min <= r_scan <= d_max):
            raise ValidationError(
                f"{system.name}: need distance_min < distance_max, with reference_distance and scan_distance inside them"
            )
        if n_x < 21 or n_r < 21:
            raise ValidationError(f"{system.name}: need n_x >= 21 and n_r >= 21")

        x = np.linspace(-x_ext, x_ext, n_x)
        r = np.linspace(d_min, d_max, n_r)
        delta = coupling * np.exp(-decay * (r - r_ref))                       # coupling at each distance
        heavy = morse(r, oo_depth, oo_alpha, oo_eq)                           # heavy-atom interaction at each distance
        e0, e1, _, _, _ = evb_adiabats(
            backend, r[None, :] / 2.0 + x[:, None], r[None, :] / 2.0 - x[:, None],
            delta[None, :], offset, depth, alpha, r_eq,
        )
        surface = e0 + heavy[None, :]

        s0, s1, weight, v_a, v_b = evb_adiabats(
            backend, r_scan / 2.0 + x, r_scan / 2.0 - x, coupling * np.exp(-decay * (r_scan - r_ref)),
            offset, depth, alpha, r_eq,
        )
        src = self.name
        shift = morse(r_scan, oo_depth, oo_alpha, oo_eq)                      # V_OO is a constant along the slice,
        obs = _slice_observables(src, x, s0 + shift, s1 + shift, weight, v_a, v_b)   # but must shift both states
        # `electronic_gap_min` (from the slice at R0) serves the rigid-scan route. The molecular route follows the
        # relaxed path instead, so it gets the smallest gap along that path, not the smallest anywhere in the table
        # (which sits at large R, far from any reactive geometry).
        relaxed_r = surface.argmin(axis=1)
        obs["electronic_gap_min_relaxed"] = Quantity(
            (e1 - e0)[np.arange(n_x), relaxed_r].min(), "eV", source=src
        )
        obs["surface_x"] = Quantity(x, "angstrom", source=src)
        obs["surface_r"] = Quantity(r, "angstrom", source=src)
        obs["surface_energy"] = Quantity(surface, "eV", source=src)
        return system.evolve(
            observables=obs,
            dynamics="E(x, R) = E0[H_el(x, R)] + V_OO(R),  H_el a 2x2 valence-bond Hamiltonian",
        )
