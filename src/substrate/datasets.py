"""Reference values from published datasets, with the provenance that lets a comparison be audited.

A computed number means little until it is set beside a measured one. This module holds the measured numbers the project compares against,
each with the entry it was read from (source, reference, method, uncertainty) and the date it was read, and turns them into `Quantity` objects
whose `source` says where they came from. The first dataset is the NIST Chemistry WebBook's gas-phase ion energetics
(`data/nist_ion_energetics.yaml`): proton affinities and acidities, the quantities that decide on which side of a hydrogen bond a proton sits.

What a comparison with them can show. The energy difference between the two wells of an asymmetric hydrogen-bonded complex, with the two heavy
atoms far apart, is the difference of the two bases' proton affinities (for cations) or of the two anions' proton affinities, the gas-phase
acidities (for anions). That difference is measured; `experimental_well_gap` returns it for each molecule template. The measured numbers are
enthalpies at 298 K and the computed ones are electronic energies of rigid fragments, so they differ by the thermal and zero-point terms and by
the relaxation of each fragment, which are not in the data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .errors import ValidationError
from .ir import Quantity
from .safeload import load_yaml

#: 1 eV in kJ/mol: N_A e / 1000, exact in the 2019 SI
KJ_PER_MOL_PER_EV = 96.48533212331002

_DATA = Path(__file__).with_name("data")


@dataclass(frozen=True)
class ReferenceValue:
    """One transcribed entry: what it is, its value as published, and where it was read."""

    id: str
    quantity: str                      # proton_affinity | acidity | acidity_gibbs
    species: str                       # the neutral molecule the entry belongs to (the key of `ReferenceSet.species`)
    value: float                       # in the set's unit (kJ/mol)
    uncertainty: float | None          # in the same unit; None where the source lists none
    method: str
    reference: str
    comment: str = ""

    def quantity_ev(self, source: str = "") -> Quantity:
        """The value in eV, with its uncertainty where there is one."""
        sigma = None if self.uncertainty is None else self.uncertainty / KJ_PER_MOL_PER_EV
        return Quantity(self.value / KJ_PER_MOL_PER_EV, "eV", sigma, source or self.reference)


@dataclass(frozen=True)
class ReferenceSet:
    source: dict
    species: dict
    values: dict[str, ReferenceValue] = field(default_factory=dict)

    def get(self, value_id: str) -> ReferenceValue:
        try:
            return self.values[value_id]
        except KeyError:
            raise ValidationError(f"no reference value '{value_id}' (have: {', '.join(self.values)})") from None

    def source_label(self, value: ReferenceValue) -> str:
        """The string a `Quantity` made from this value carries as its source."""
        return f"{self.source['database']}: {value.reference} ({value.method}), retrieved {self.source['retrieved']}"

    def quantity(self, value_id: str) -> Quantity:
        value = self.get(value_id)
        return value.quantity_ev(self.source_label(value))

    def gap(self, weaker_id: str, stronger_id: str) -> Quantity:
        """How much more tightly the stronger base holds the proton than the weaker, in eV: the difference of the two entries, which must be of
        the same kind (two proton affinities, or two acidities). Zero, with zero uncertainty, when both are the same entry. The uncertainty is
        the two entries' in quadrature, and is None if either lists none (it is not guessed)."""
        weaker, stronger = self.get(weaker_id), self.get(stronger_id)
        if weaker.quantity != stronger.quantity:
            raise ValidationError(f"cannot take the gap between a {weaker.quantity} and a {stronger.quantity}")
        if weaker_id == stronger_id:
            return Quantity(0.0, "eV", 0.0, self.source_label(weaker))
        sigma = (None if weaker.uncertainty is None or stronger.uncertainty is None
                 else math.hypot(weaker.uncertainty, stronger.uncertainty) / KJ_PER_MOL_PER_EV)
        return Quantity((stronger.value - weaker.value) / KJ_PER_MOL_PER_EV, "eV", sigma,
                        f"{self.source['database']}: {stronger.reference} and {weaker.reference}, retrieved {self.source['retrieved']}")


@lru_cache(maxsize=None)
def load_reference_set(name: str = "nist_ion_energetics") -> ReferenceSet:
    """Read a reference set from the package's `data/` directory."""
    path = _DATA / f"{name}.yaml"
    if not path.exists():
        raise ValidationError(f"no reference set '{name}' (have: {', '.join(sorted(p.stem for p in _DATA.glob('*.yaml')))})")
    raw = load_yaml(path.read_text(encoding="utf-8"), path.name)
    values = {}
    for entry in raw["values"]:
        if entry["species"] not in raw["species"]:
            raise ValidationError(f"{name}: value '{entry['id']}' names an unknown species '{entry['species']}'")
        if entry["id"] in values:
            raise ValidationError(f"{name}: duplicate value id '{entry['id']}'")
        values[entry["id"]] = ReferenceValue(
            id=entry["id"], quantity=entry["quantity"], species=entry["species"], value=float(entry["value"]),
            uncertainty=None if entry.get("uncertainty") is None else float(entry["uncertainty"]),
            method=str(entry.get("method", "")), reference=str(entry.get("reference", "")), comment=str(entry.get("comment", "")))
    return ReferenceSet(source=raw["source"], species=raw["species"], values=values)


#: For each molecule template, the entries of the weaker base (the donor side, whose well is the higher) and the stronger one (the acceptor
#: side); the element each binds the proton with is the species basic_atom in the data file. A symmetric template names one entry twice:
#: its two wells are equal.
TEMPLATE_BASES: dict[str, tuple[str, str]] = {
    "zundel_cation": ("pa_h2o", "pa_h2o"),
    "ammonium_dimer_cation": ("pa_nh3", "pa_nh3"),
    "bifluoride_anion": ("acidity_hf", "acidity_hf"),
    "water_ammonia_cation": ("pa_h2o", "pa_nh3"),
    "methanol_water_cation": ("pa_h2o", "pa_ch3oh"),
    "ammonia_methylamine_cation": ("pa_nh3", "pa_ch3nh2"),
    "fluoride_methanol_anion": ("acidity_hf", "acidity_ch3oh"),
    "chloride_hf_anion": ("acidity_hcl", "acidity_hf"),
}


def experimental_well_gap(template: str, reference_set: ReferenceSet | None = None) -> Quantity:
    """The measured energy by which the acceptor-side well of a template's complex lies below the donor-side one, far from the other atom:
    the difference of the two bases' proton affinities (cations) or gas-phase acidities (anions)."""
    if template not in TEMPLATE_BASES:
        raise ValidationError(f"no measured proton-affinity pair for the molecule template '{template}' (have: {', '.join(TEMPLATE_BASES)})")
    reference_set = reference_set or load_reference_set()
    weaker, stronger = TEMPLATE_BASES[template]
    return reference_set.gap(weaker, stronger)
