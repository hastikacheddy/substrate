"""Electronic structure -> molecular: hand the computed 2D surface to a molecule with atoms and masses.

Unlike the route into the quantum scale (a 1D slice with a frozen heavy-atom distance), this keeps the heavy-atom
distance as a coordinate, so the molecular scale sees the relaxed surface. Masses are applied here: the surface is
mass-independent (Born-Oppenheimer), so H and D share one translated surface.
"""
from __future__ import annotations

import numpy as np

from ..base import Translator, ValidationIssue
from ..engines.electronic import KIND_2D
from ..engines.qc_scan import KIND as QC_KIND_2D
from ..engines.molecular import KIND as MOLECULAR_KIND
from ..ir import Quantity, Scale, ScientificSystem
from .electronic_to_quantum import MIN_GAP_EV

NAME = "electronic_to_molecular.surface_handoff"


class ElectronicToMolecular(Translator):
    name = NAME
    source = Scale.ELECTRONIC_STRUCTURE
    source_kinds = (KIND_2D, QC_KIND_2D)
    target = Scale.MOLECULAR
    target_kind = MOLECULAR_KIND
    approximations = (
        "Born-Oppenheimer: the nuclei move on the electronic ground-state surface; non-adiabatic effects are neglected",
        "the surface is independent of nuclear mass (same surface for H, D and T)",
        "collinear heavy atom - proton - heavy atom geometry: the two heavy atoms share one mass",
        "the surface is carried as a table; its resolution and range bound the accuracy of the molecular analysis",
    )

    def validate(self, system: ScientificSystem) -> list[ValidationIssue]:
        x = system.obs("surface_x", "angstrom", default=None)
        r = system.obs("surface_r", "angstrom", default=None)
        e = system.obs("surface_energy", "eV", default=None)
        if x is None or r is None or e is None:
            return [ValidationIssue("error", "system has not been solved (surface_x/surface_r/surface_energy missing)")]
        for name in ("particle_mass", "heavy_atom_mass", "temperature"):
            value = system.param(name, default=None)
            if value is None or value <= 0:
                return [ValidationIssue("error", f"a positive '{name}' parameter is required")]
        if e.shape != (len(x), len(r)) or not np.all(np.isfinite(e)):
            return [ValidationIssue("error", "surface_energy must be a finite (len(surface_x), len(surface_r)) table")]
        gap = system.obs("electronic_gap_min_relaxed", "eV", default=None)
        if gap is not None and gap < MIN_GAP_EV:
            return [ValidationIssue(
                "warning",
                f"minimum electronic gap along the relaxed path is {gap:.2f} eV, below ~{MIN_GAP_EV} eV: non-adiabatic effects (neglected) may matter",
            )]
        return []

    def translate(self, system: ScientificSystem) -> ScientificSystem:
        src = self.name
        heavy, light = system.parameters["heavy_atom_mass"], system.parameters["particle_mass"]
        temperature = system.parameters["temperature"]
        return ScientificSystem(
            name=f"{system.name} [molecular]",
            scale=Scale.MOLECULAR,
            kind=MOLECULAR_KIND,
            parameters={
                "mass_donor": Quantity(heavy.value, "amu", heavy.sigma, src),
                "mass_acceptor": Quantity(heavy.value, "amu", heavy.sigma, src),
                "mass_hydrogen": Quantity(light.value, "amu", light.sigma, src),
                "temperature": Quantity(temperature.value, "K", temperature.sigma, temperature.source),
                "surface_x": Quantity(system.obs("surface_x"), "angstrom", source=src),
                "surface_r": Quantity(system.obs("surface_r"), "angstrom", source=src),
                "surface_energy": Quantity(system.obs("surface_energy"), "eV", source=src),
            },
            structure={"derivation": {"from": system.name, "surface": "electronic ground state, 2D (x, R)"}},
            dynamics="atoms on E(x, R)",
        )
