"""The separated fragments of a proton-transfer complex.

With the two heavy atoms far apart, the energy difference between the proton sitting on one side and on the other is the difference of two
bases' proton affinities (a cation) or of two acids' gas-phase acidities (an anion). It is a sum over four fragments: each side with and
without the proton. `separated_fragments` builds those four from a molecule, started in the geometry each has inside the complex, so the
fragments can be relaxed or given a thermochemical correction; `exchange` combines four values into the difference.
"""
from __future__ import annotations

from .diatomics import DIATOMICS
from .engines.qc_scan import build_geometry

Atoms = tuple[tuple[str, float, float, float], ...]
Fragment = tuple[Atoms, int]                                  # atoms (angstrom) and total charge


def separated_fragments(molecule: dict, separation: float = 10.0) -> dict[tuple[str, bool], Fragment]:
    """{(end, holds the proton): (atoms, charge)} for the donor side and the acceptor side, each with and without the proton, the heavy atoms
    `separation` angstrom apart and a proton at the free diatomic's bond length from its own heavy atom. A cation's charge sits on the fragment
    that holds the proton; an anion's sits on the one that does not. A neutral complex is not supported: it has no ion to put the charge on."""
    atoms = molecule["atoms"]
    donor, acceptor, proton = molecule["donor"], molecule["acceptor"], molecule["proton"]
    charge = int(molecule.get("charge", 0))
    if charge == 0:
        raise ValueError("separated fragments are defined for an ion (a charged complex), not a neutral one")
    groups = {"donor": [donor, *molecule["donor_group"]], "acceptor": [acceptor, *molecule["acceptor_group"]]}
    fragments = {}
    for end, sign in (("donor", -1.0), ("acceptor", +1.0)):
        heavy = atoms[groups[end][0]][0]
        geometry = build_geometry(molecule, sign * (separation / 2.0 - DIATOMICS[heavy].r_e), separation)
        for holds_proton in (True, False):
            members = groups[end] + ([proton] if holds_proton else [])
            on_this_fragment = holds_proton if charge > 0 else not holds_proton
            fragments[(end, holds_proton)] = (tuple(geometry[i] for i in members), charge if on_this_fragment else 0)
    return fragments


def exchange(values: dict) -> float:
    """value(proton on the donor side) - value(proton on the acceptor side), from the four fragments' values: the two protonation states of the pair."""
    return (values[("donor", True)] + values[("acceptor", False)]) - (values[("donor", False)] + values[("acceptor", True)])
