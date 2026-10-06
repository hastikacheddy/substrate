"""Electronic-structure scale, for real: a 2D proton-transfer surface from quantum-chemistry single-point energies.

Takes a molecule with a donor-H-acceptor axis and computes E(x, R) on a grid with an ab initio (or DFT) program:

    x   displacement of the proton from the midpoint of the two heavy atoms (angstrom)
    R   heavy-atom distance (angstrom)

Every other atom belongs to a rigid group attached to the donor or to the acceptor and moves with its heavy atom. So the
scan is *rigid*: bond lengths and angles inside the groups stay at their reference values, and nothing relaxes except through
the two scanned coordinates. It produces exactly the observables of the model engines, so everything downstream (the quantum,
molecular, reaction, biophysical and biological scales) works unchanged on real chemistry.

Structure (non-numeric configuration):

    molecule: {atoms: [[symbol, x, y, z], ...],  donor, acceptor, proton: atom indices,
               donor_group, acceptor_group: indices of the atoms that travel with each heavy atom,
               charge, spin (2S)}
    method:   {theory: hf | mp2 | dft:<functional>,  basis: <basis name>,  program: pyscf}
    scan:     {mirror_symmetric: true}    # E(x, R) = E(-x, R): compute x >= 0 only (checked, not assumed)

The reference geometry must have the donor, proton and acceptor on the z axis with the donor at lower z.
"""
from __future__ import annotations

import numpy as np
from scipy import constants

from ..backends import SolverBackend
from ..base import Engine
from ..errors import ValidationError
from ..ir import Quantity, Scale, ScientificSystem
from ..pes import MIN_PROMINENCE_EV, locate_wells, prominent_minima
from ..molecules import resolve_molecule
from ..qc import QCCache, QCJob, compute_cached, get_program

KIND = "electronic.qc_scan_2d"
HARTREE_EV = constants.physical_constants["Hartree energy in eV"][0]
#: a declared mirror symmetry is verified on one pair of points; the two energies must agree to this (hartree)
MIRROR_TOLERANCE = 1e-6


def build_geometry(molecule: dict, x: float, r: float) -> tuple[tuple[str, float, float, float], ...]:
    """Atoms for proton displacement x and heavy-atom distance r. The donor (and its group) go to z = -r/2, the acceptor
    (and its group) to z = +r/2, the proton to z = x on the axis; groups move rigidly."""
    atoms = molecule["atoms"]
    donor, acceptor, proton = molecule["donor"], molecule["acceptor"], molecule["proton"]
    shift = {i: 0.0 for i in range(len(atoms))}
    for i in [donor, *molecule["donor_group"]]:
        shift[i] = -r / 2.0 - atoms[donor][3]
    for i in [acceptor, *molecule["acceptor_group"]]:
        shift[i] = r / 2.0 - atoms[acceptor][3]
    out = []
    for i, (symbol, ax, ay, az) in enumerate(atoms):
        z = x if i == proton else az + shift[i]
        out.append((symbol, float(ax), float(ay), float(z)))
    return tuple(out)


def validate_molecule(name: str, molecule: dict) -> None:
    atoms = molecule["atoms"]
    n = len(atoms)
    roles = {"donor": [molecule["donor"]], "acceptor": [molecule["acceptor"]], "proton": [molecule["proton"]],
             "donor_group": list(molecule["donor_group"]), "acceptor_group": list(molecule["acceptor_group"])}
    seen: dict[int, str] = {}
    for role, indices in roles.items():
        for i in indices:
            if not isinstance(i, int) or not 0 <= i < n:
                raise ValidationError(f"{name}: {role} refers to atom {i}, but the molecule has {n} atoms")
            if i in seen:
                raise ValidationError(f"{name}: atom {i} is listed as both {seen[i]} and {role}")
            seen[i] = role
    unassigned = sorted(set(range(n)) - set(seen))
    if unassigned:
        raise ValidationError(
            f"{name}: atoms {unassigned} are in no group: every atom must be the donor, the acceptor, the proton, or "
            f"in the donor or acceptor group (otherwise it would not move when the heavy atoms do)")
    for role in ("donor", "acceptor", "proton"):
        _, x, y, _ = atoms[roles[role][0]]
        if abs(x) > 1e-6 or abs(y) > 1e-6:
            raise ValidationError(f"{name}: the {role} must lie on the z axis in the reference geometry (x={x:g}, y={y:g})")
    if not atoms[molecule["donor"]][3] < atoms[molecule["acceptor"]][3]:
        raise ValidationError(f"{name}: the donor must have the lower z coordinate in the reference geometry")


class QCScanEngine(Engine):
    name = KIND
    scale = Scale.ELECTRONIC_STRUCTURE
    kinds = (KIND,)
    approximations = (
        "single-point energies at the stated level of theory and basis set: the surface is only as good as that method",
        "rigid scan: every atom other than the proton moves with its heavy atom as a frozen group, so internal coordinates of "
        "the groups do not relax in response to the proton or to the heavy-atom distance",
        "collinear donor-proton-acceptor axis: no bending of the hydrogen bond",
        "gas-phase energies: no solvent, no enzyme environment",
        "Born-Oppenheimer ground state; the 'electronic gap' reported is the HOMO-LUMO orbital gap, a proxy for the true "
        "excitation gap (overestimated by Hartree-Fock)",
    )

    def solve(self, system: ScientificSystem, backend: SolverBackend) -> ScientificSystem:
        self.check_kind(system)
        structure = system.structure
        for key in ("molecule", "method"):
            if key not in structure:
                raise ValidationError(f"{system.name}: structure needs a '{key}' section")
        try:
            molecule, method = resolve_molecule(structure["molecule"]), structure["method"]
        except ValueError as error:
            raise ValidationError(f"{system.name}: {error}") from None
        try:
            validate_molecule(system.name, molecule)
        except KeyError as missing:
            raise ValidationError(f"{system.name}: molecule is missing {missing}") from None

        x_ext = system.param("x_extent", "angstrom", default=0.7)
        d_min = system.param("distance_min", "angstrom", default=2.3)
        d_max = system.param("distance_max", "angstrom", default=3.0)
        r_scan = system.param("scan_distance", "angstrom")
        n_x = int(system.param("n_x", default=29))
        n_r = int(system.param("n_r", default=15))
        if min(x_ext, d_min) <= 0 or not d_min < d_max or not d_min <= r_scan <= d_max:
            raise ValidationError(
                f"{system.name}: need positive x_extent and distance_min < distance_max, with scan_distance inside them")
        if n_x < 9 or n_x % 2 == 0 or n_r < 3:
            raise ValidationError(f"{system.name}: n_x must be odd and >= 9, n_r >= 3 (the grid must contain x = 0)")
        mirror = bool(structure.get("scan", {}).get("mirror_symmetric", False))

        program = get_program(method.get("program", "pyscf"))
        charge, spin = int(molecule.get("charge", 0)), int(molecule.get("spin", 0))
        theory, basis = method["theory"], method["basis"]

        x = np.linspace(-x_ext, x_ext, n_x)
        r = np.linspace(d_min, d_max, n_r)
        compute_x = x[x >= -1e-12] if mirror else x                    # with mirror symmetry only x >= 0 is computed
        jobs: list[QCJob] = []
        index: dict[tuple[str, int, int], int] = {}
        for i, xi in enumerate(compute_x):
            for j, rj in enumerate(r):
                index[("surface", i, j)] = len(jobs)
                jobs.append(QCJob(build_geometry(molecule, float(xi), float(rj)), charge, spin, theory, basis))
            index[("slice", i, 0)] = len(jobs)
            jobs.append(QCJob(build_geometry(molecule, float(xi), r_scan), charge, spin, theory, basis))
        mirror_job = None
        if mirror:                                                      # verify the declared symmetry on one pair of points
            mirror_job = len(jobs)
            jobs.append(QCJob(build_geometry(molecule, -float(x[-1]), r_scan), charge, spin, theory, basis))

        results, n_computed, n_cached = compute_cached(program, jobs, QCCache())
        bad = [(jobs[k], res) for k, res in enumerate(results) if not res.converged]
        if bad:
            raise ValidationError(
                f"{system.name}: SCF did not converge for {len(bad)} of {len(jobs)} geometries "
                f"(first: proton z = {bad[0][0].atoms[molecule['proton']][3]:.3f} angstrom)")

        energy = np.array([res.energy_hartree for res in results])
        gap = np.array([np.nan if res.homo_lumo_gap_hartree is None else res.homo_lumo_gap_hartree for res in results])
        if mirror:
            top = index[("slice", len(compute_x) - 1, 0)]
            difference = abs(energy[mirror_job] - energy[top])
            if difference > MIRROR_TOLERANCE:
                raise ValidationError(
                    f"{system.name}: scan.mirror_symmetric is set, but E(+x) and E(-x) differ by {difference:.3g} hartree: "
                    f"the molecule is not symmetric under exchanging donor and acceptor")

        def assemble(kind: str, columns: int, values: np.ndarray) -> np.ndarray:
            half = np.array([[values[index[(kind, i, j)]] for j in range(columns)] for i in range(len(compute_x))])
            if not mirror:
                return half
            negative = half[1:][::-1]                                   # x < 0 mirrors x > 0
            return np.vstack([negative, half])

        surface_ha, slice_ha = assemble("surface", n_r, energy), assemble("slice", 1, energy)[:, 0]
        surface_gap, slice_gap = assemble("surface", n_r, gap), assemble("slice", 1, gap)[:, 0]
        reference = min(surface_ha.min(), slice_ha.min())
        surface = (surface_ha - reference) * HARTREE_EV
        scan = (slice_ha - reference) * HARTREE_EV
        relaxed_r = surface.argmin(axis=1)

        src = f"{self.name}:{program.name}"
        obs = {
            "surface_x": Quantity(x, "angstrom", source=src),
            "surface_r": Quantity(r, "angstrom", source=src),
            "surface_energy": Quantity(surface, "eV", source=src),
            "surface_gap": Quantity(surface_gap * HARTREE_EV, "eV", source=src),
            "scan_coordinate": Quantity(x, "angstrom", source=src),
            "scan_energy": Quantity(scan, "eV", source=src),
            "electronic_gap_min": Quantity(np.nanmin(slice_gap) * HARTREE_EV, "eV", source=src),
            "electronic_gap_min_relaxed": Quantity(
                np.nanmin(surface_gap[np.arange(n_x), relaxed_r]) * HARTREE_EV, "eV", source=src),
            "n_minima": Quantity(len(prominent_minima(scan, MIN_PROMINENCE_EV)), "count", source=src),
            "reference_energy": Quantity(reference, "hartree", source=src),
            "calculations_run": Quantity(n_computed, "count", source=src),
            "calculations_cached": Quantity(n_cached, "count", source=src),
        }
        wells = locate_wells(scan)
        if wells is not None:
            i_left, i_top, i_right = wells
            obs["classical_barrier"] = Quantity(scan[i_top] - scan[i_left], "eV", source=src)
            obs["classical_reaction_energy"] = Quantity(scan[i_right] - scan[i_left], "eV", source=src)
        return system.evolve(
            observables=obs,
            structure={**structure, "molecule": molecule},          # the resolved molecule, so the solved system records its atoms
            dynamics=f"{theory}/{basis} single-point energies on a rigid-group (x, R) grid; E(x, R) in eV above the minimum",
        )
