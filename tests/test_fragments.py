"""The four separated fragments of a proton-transfer complex: they must be the complex taken apart, with nothing gained or lost."""
from collections import Counter

import numpy as np
import pytest

from substrate.datasets import TEMPLATE_BASES
from substrate.diatomics import DIATOMICS
from substrate.fragments import exchange, separated_fragments
from substrate.molecules import resolve_molecule


def _formula(atoms):
    return Counter(symbol for symbol, *_ in atoms)


@pytest.mark.parametrize("template", sorted(TEMPLATE_BASES))
def test_each_protonation_state_is_the_whole_complex_in_atoms_and_in_charge(template):
    molecule = resolve_molecule({"template": template})
    fragments = separated_fragments(molecule)
    whole = _formula(molecule["atoms"])
    for donor_side, acceptor_side in ((("donor", True), ("acceptor", False)), (("donor", False), ("acceptor", True))):
        (a, qa), (b, qb) = fragments[donor_side], fragments[acceptor_side]
        assert _formula(a) + _formula(b) == whole, (template, donor_side)                     # no atom lost or duplicated
        assert qa + qb == molecule["charge"], (template, donor_side)                          # the complex's charge is on one of the two
    for (end, holds_proton), (atoms, charge) in fragments.items():
        assert charge == (molecule["charge"] if holds_proton == (molecule["charge"] > 0) else 0), (template, end, holds_proton)   # a cation's charge rides on the proton, an anion's does not


@pytest.mark.parametrize("template", sorted(TEMPLATE_BASES))
def test_the_two_sides_are_the_heavy_atom_distance_apart_and_the_proton_sits_at_the_bond_length_of_its_own_heavy_atom(template):
    molecule = resolve_molecule({"template": template})
    separation = 10.0
    fragments = separated_fragments(molecule, separation)
    donor_atom, acceptor_atom = fragments[("donor", False)][0][0], fragments[("acceptor", False)][0][0]       # the heavy atom is listed first
    assert abs(acceptor_atom[3] - donor_atom[3]) == pytest.approx(separation, abs=1e-9)
    for end, heavy in (("donor", donor_atom), ("acceptor", acceptor_atom)):
        with_proton = fragments[(end, True)][0]
        distance = np.linalg.norm(np.array(with_proton[-1][1:]) - np.array(heavy[1:]))                         # the proton is listed last
        assert distance == pytest.approx(DIATOMICS[heavy[0]].r_e, abs=1e-6), (template, end)
        assert fragments[(end, False)][0] == with_proton[:-1]                                                  # taking the proton away changes nothing else


def test_a_symmetric_complex_gives_the_same_fragments_on_both_sides_and_a_neutral_one_is_refused():
    zundel = resolve_molecule({"template": "zundel_cation"})
    fragments = separated_fragments(zundel)
    assert _formula(fragments[("donor", True)][0]) == _formula(fragments[("acceptor", True)][0])
    assert fragments[("donor", True)][1] == fragments[("acceptor", True)][1] == zundel["charge"]
    assert fragments == separated_fragments(zundel, 10.0)                                     # the documented default: 10 angstrom, as the cached relaxations assume
    with pytest.raises(ValueError, match="ion"):
        separated_fragments({**zundel, "charge": 0})


def test_exchange_is_the_difference_of_the_two_protonation_states():
    values = {("donor", True): 1.0, ("donor", False): 10.0, ("acceptor", True): 100.0, ("acceptor", False): 1000.0}
    assert exchange(values) == (1.0 + 1000.0) - (10.0 + 100.0) == 891.0
    swapped = {("donor", True): 100.0, ("donor", False): 1000.0, ("acceptor", True): 1.0, ("acceptor", False): 10.0}          # the two sides' labels exchanged
    assert exchange(swapped) == -891.0
    assert exchange({("donor", True): 3.0, ("donor", False): 7.0, ("acceptor", True): 3.0, ("acceptor", False): 7.0}) == 0.0    # a symmetric pair has no gap
    assert exchange({k: v + 5.0 for k, v in values.items()}) == 891.0                          # a common offset in the four values cancels
