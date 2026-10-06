"""Electronic structure -> quantum: hand the computed Born-Oppenheimer surface to nuclear quantum dynamics.

The electrons are integrated out: the nucleus moves on E0(x). Because the surface does not depend on nuclear
mass, the same translated surface serves protium, deuterium and tritium; the mass is applied here, from the
electronic system's `particle_mass` context parameter.
"""
from __future__ import annotations

import numpy as np

from ..base import Translator, ValidationIssue
from ..ir import Quantity, Scale, ScientificSystem
from ..pes import locate_wells, prominent_minima
from ..engines.quantum import KIND_TABULATED

NAME = "electronic_to_quantum.born_oppenheimer"

#: heuristic floor for the electronic gap, a bit above one O-H stretching quantum (~0.4 eV)
MIN_GAP_EV = 0.5
#: the table must end this far (eV) above the barrier top, or wavefunctions leak off its edges
MIN_EDGE_MARGIN_EV = 0.5
#: fewest scan points the nuclear solver can use at all (it splines the table), and the spacing above which the spline is
#: being asked to resolve structure the table does not contain: a proton's ground-state wavefunction is ~0.1 angstrom wide
MIN_SCAN_POINTS = 15
MAX_SPACING = 0.1


class ElectronicToQuantum(Translator):
    name = NAME
    source = Scale.ELECTRONIC_STRUCTURE
    target = Scale.QUANTUM
    target_kind = KIND_TABULATED
    approximations = (
        "Born-Oppenheimer: the nucleus moves on the electronic ground-state surface; non-adiabatic transitions "
        "and geometric-phase effects are neglected",
        "the surface is independent of nuclear mass (same surface for H, D and T)",
        "the scanned coordinate is the only nuclear degree of freedom; its mass is the bare particle mass",
        "the surface is carried as a table and splined by the quantum engine, so scan resolution limits accuracy",
    )

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        x = system.obs("scan_coordinate", "angstrom", default=None)
        e = system.obs("scan_energy", "eV", default=None)
        if x is None or e is None:
            return [ValidationIssue("error", "system has not been solved (scan_coordinate/scan_energy missing)")]
        mass = system.param("particle_mass", "amu", default=None)
        if mass is None or mass <= 0:
            return [ValidationIssue("error", "a positive 'particle_mass' parameter (amu) is required")]
        if len(x) < MIN_SCAN_POINTS or not np.all(np.diff(x) > 0) or not np.all(np.isfinite(e)):
            return [ValidationIssue(
                "error", f"scan must be >= {MIN_SCAN_POINTS} finite points on a strictly increasing coordinate")]

        n_wells = len(prominent_minima(e))
        if n_wells < 2:
            return [ValidationIssue(
                "error",
                f"the ground-state surface has {n_wells} well: the proton is not localised on either heavy atom, "
                f"so there is no reactant/product pair to translate",
            )]
        issues = []
        spacing = float(np.max(np.diff(x)))
        if spacing > MAX_SPACING:
            issues.append(ValidationIssue(
                "warning",
                f"the scan spacing ({spacing:.2f} angstrom) is coarse next to a proton's zero-point width (~0.1 angstrom): "
                f"the nuclear levels depend on how the spline interpolates between points"))
        if n_wells > 2:
            issues.append(ValidationIssue(
                "warning", f"the surface has {n_wells} wells; only the two lowest are treated as reactant and product"
            ))
        gap = system.obs("electronic_gap_min", "eV", default=None)
        if gap is not None and gap < MIN_GAP_EV:
            issues.append(ValidationIssue(
                "warning",
                f"minimum electronic gap is {gap:.2f} eV, below ~{MIN_GAP_EV} eV: the excited electronic state is "
                f"close enough that non-adiabatic effects (neglected here) may matter",
            ))
        i_left, i_top, i_right = locate_wells(e)
        edge = min(e[0], e[-1]) - e[i_top]
        if edge < MIN_EDGE_MARGIN_EV:
            issues.append(ValidationIssue(
                "warning",
                f"the scan edges are only {edge:.2f} eV above the barrier top (< {MIN_EDGE_MARGIN_EV} eV): "
                f"wavefunctions may leak off the end of the table",
            ))
        return issues

    def translate(self, system: ScientificSystem) -> ScientificSystem:
        src = self.name
        mass = system.parameters["particle_mass"]
        params = {
            "mass": Quantity(mass.value, "amu", mass.sigma, src),
            "pes_x": Quantity(system.obs("scan_coordinate"), "angstrom", source=src),
            "pes_energy": Quantity(system.obs("scan_energy"), "eV", source=src),
        }
        temperature = system.parameters.get("temperature")
        if temperature is not None:
            params["temperature"] = Quantity(temperature.value, "K", temperature.sigma, temperature.source)
        return ScientificSystem(
            name=f"{system.name} [quantum]",
            scale=Scale.QUANTUM,
            kind=KIND_TABULATED,
            parameters=params,
            structure={"derivation": {"from": system.name, "surface": "electronic ground state"}},
            dynamics="(-hbar^2/2m d^2/dx^2 + E0(x)) psi = E psi",
        )
