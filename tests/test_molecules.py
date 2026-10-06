"""Molecule templates: the geometry is checked against numbers worked out independently of the code that builds it."""
import math

import numpy as np
import pytest

from substrate import Quantity, Scale, ScientificSystem, ValidationError
from substrate.engines.qc_scan import QCScanEngine, build_geometry, validate_molecule
from substrate.molecules import (
    HEAVY_ATOM_MASS, TEMPLATES, ammonia_methylamine_cation, ammonium_dimer_cation, bifluoride_anion, chloride_hf_anion, fluoride_methanol_anion,
    methanol_water_cation, resolve_molecule, water_ammonia_cation, zundel_cation,
)
from substrate.qc import QCProgram, QCResult, register_program


def positions(molecule):
    return np.array([a[1:] for a in molecule["atoms"]], dtype=float)


def symbols(molecule):
    return [a[0] for a in molecule["atoms"]]


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_is_a_valid_molecule_for_the_engine(name):
    molecule = TEMPLATES[name]()
    validate_molecule(name, molecule)
    assert name in HEAVY_ATOM_MASS


ASYMMETRIC = {"water_ammonia_cation", "methanol_water_cation", "ammonia_methylamine_cation", "fluoride_methanol_anion", "chloride_hf_anion"}                       # asymmetric on purpose
SYMMETRIC = sorted(set(TEMPLATES) - ASYMMETRIC)


def exchange_operations(molecule):
    """The turns and vertical reflections (angles in degrees, every half degree) that, together with z -> -z, map the atoms onto themselves."""
    pos, sym = positions(molecule), symbols(molecule)

    def maps_onto_itself(matrix):
        image = np.column_stack([pos[:, :2] @ matrix.T, -pos[:, 2]])
        return all(any(sym[j] == sym[i] and np.allclose(pos[j], image[i], atol=1e-9) for j in range(len(pos))) for i in range(len(pos)))

    def turn(a):
        a = math.radians(a)
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def reflection(b):                                   # across the vertical plane through the axis at azimuth b: azimuth a -> 2b - a
        b = math.radians(b)
        return np.array([[math.cos(2 * b), math.sin(2 * b)], [math.sin(2 * b), -math.cos(2 * b)]])

    angles = np.arange(0.0, 360.0, 0.5)
    return [a for a in angles if maps_onto_itself(turn(a))], [b for b in angles if maps_onto_itself(reflection(b))]


@pytest.mark.parametrize("name", SYMMETRIC)
def test_every_symmetric_template_maps_onto_itself_with_donor_and_acceptor_exchanged(name):
    """Sending z -> -z together with SOME operation on (x, y) (a turn or a reflection) must map the atoms onto themselves:
    then the energy with the proton at +x equals that at -x, which is what `mirror_symmetric` relies on. Zundel's staggered waters
    need a reflection across the vertical plane at 45 degrees (a C2 axis across the O-O line); the ammonium dimer a 60 degree turn
    (or inversion, 180); the bare FHF- anything."""
    turns, reflections = exchange_operations(TEMPLATES[name]())
    assert turns or reflections, f"{name}: no operation on (x, y) completes the donor-acceptor exchange"
    if name == "zundel_cation":
        assert 45.0 in reflections and 0.0 not in turns and 180.0 not in turns       # not a rotation: the two waters are staggered
    if name == "ammonium_dimer_cation":
        assert 60.0 in turns and 180.0 in turns


@pytest.mark.parametrize("name", sorted(ASYMMETRIC))
def test_the_asymmetric_templates_have_no_exchange_operation(name):
    assert exchange_operations(TEMPLATES[name]()) == ([], [])                        # so a mirror_symmetric claim for them is false by construction


def test_the_ammonium_dimers_flank_geometry_is_what_its_parameters_say():
    molecule = ammonium_dimer_cation(r_nh=1.02, theta_c=110.0)
    pos = positions(molecule)
    assert symbols(molecule) == ["N", "N", "H"] + ["H"] * 6 and molecule["charge"] == 1
    for n_index, group, towards in ((0, molecule["donor_group"], +1), (1, molecule["acceptor_group"], -1)):
        assert len(group) == 3
        for h in group:
            bond = pos[h] - pos[n_index]
            assert np.linalg.norm(bond) == pytest.approx(1.02, abs=1e-12)                   # N-H length
            axis = np.array([0.0, 0.0, towards])                                              # from this N towards the shared proton
            assert math.degrees(math.acos(bond @ axis / np.linalg.norm(bond))) == pytest.approx(110.0, abs=1e-9)
        azimuths = sorted(math.degrees(math.atan2(*pos[h][:2][::-1])) % 360 for h in group)
        assert np.diff(azimuths + [azimuths[0] + 360]) == pytest.approx([120.0] * 3, abs=1e-9)    # three-fold: 120 degrees apart
    donor_azimuth = math.degrees(math.atan2(pos[3][1], pos[3][0]))
    acceptor_azimuths = [math.degrees(math.atan2(pos[h][1], pos[h][0])) for h in molecule["acceptor_group"]]
    offsets = sorted(((a - donor_azimuth) % 120) for a in acceptor_azimuths)
    assert offsets == pytest.approx([60.0] * 3, abs=1e-9)                                   # staggered: halfway between the donor's hydrogens


def test_the_bifluoride_is_a_bare_collinear_triatomic():
    molecule = bifluoride_anion()
    assert symbols(molecule) == ["F", "F", "H"] and molecule["charge"] == -1
    assert molecule["donor_group"] == [] and molecule["acceptor_group"] == []
    assert np.allclose(positions(molecule)[:, :2], 0.0)                                      # everything on the axis


def test_the_options_change_the_geometry():
    assert np.linalg.norm(positions(ammonium_dimer_cation(r_nh=1.10))[3] - positions(ammonium_dimer_cation(r_nh=1.10))[0]) == pytest.approx(1.10)
    assert not np.allclose(positions(ammonium_dimer_cation(theta_c=100.0)), positions(ammonium_dimer_cation(theta_c=110.0)))


def test_the_scan_geometry_carries_each_group_with_its_heavy_atom():
    molecule = ammonium_dimer_cation()
    reference, scanned = positions(molecule), np.array([a[1:] for a in build_geometry(molecule, 0.31, 2.9)])
    assert scanned[2] == pytest.approx([0.0, 0.0, 0.31])                                      # the proton sits at x
    assert scanned[0][2] == pytest.approx(-1.45) and scanned[1][2] == pytest.approx(1.45)     # the heavy atoms at -+R/2
    for group, heavy in ((molecule["donor_group"], 0), (molecule["acceptor_group"], 1)):
        for h in group:
            assert scanned[h] - scanned[heavy] == pytest.approx(reference[h] - reference[heavy], abs=1e-12)   # rigid with its heavy atom


# -- resolving a named template ------------------------------------------------------------------------------------------------------
def test_a_named_template_resolves_to_the_full_structure_and_remembers_its_name():
    resolved = resolve_molecule({"template": "bifluoride_anion"})
    assert resolved["template"] == "bifluoride_anion" and resolved["atoms"] == bifluoride_anion()["atoms"]
    assert resolve_molecule(resolved) == resolved                                            # resolving a resolved spec changes nothing


def test_explicit_keys_and_options_override_the_template():
    assert resolve_molecule({"template": "zundel_cation", "charge": 2})["charge"] == 2
    short = resolve_molecule({"template": "ammonium_dimer_cation", "options": {"r_nh": 1.10}})
    assert short["atoms"] == ammonium_dimer_cation(r_nh=1.10)["atoms"] and "options" not in short


def test_a_spec_that_lists_atoms_passes_through_untouched():
    spec = zundel_cation()
    assert resolve_molecule(spec) is spec


def test_an_unknown_template_names_the_choices():
    with pytest.raises(ValueError, match="unknown molecule template 'water' .*zundel_cation"):
        resolve_molecule({"template": "water"})


# -- the engine ------------------------------------------------------------------------------------------------------------------------
class _FakeProgram(QCProgram):
    """A stand-in that returns a harmonic well in the proton coordinate, so engine plumbing is tested without PySCF."""
    name = "fake-molecules"

    def compute(self, jobs):
        return [QCResult(0.5 * job.atoms[2][3] ** 2, True, 0.1) for job in jobs]          # the proton is atom 2 in every template


def _system(molecule):
    p = {"scan_distance": (2.7, "angstrom"), "x_extent": (0.5, "angstrom"), "distance_min": (2.5, "angstrom"),
         "distance_max": (2.9, "angstrom"), "n_x": (9, "1"), "n_r": (3, "1"), "particle_mass": (1.007276, "amu"),
         "heavy_atom_mass": (14.0, "amu"), "temperature": (300.0, "K")}
    return ScientificSystem("t", Scale.ELECTRONIC_STRUCTURE, "electronic.qc_scan_2d",
                            parameters={k: Quantity(v, u) for k, (v, u) in p.items()},
                            structure={"molecule": molecule, "method": {"theory": "hf", "basis": "sto-3g", "program": "fake-molecules"},
                                       "scan": {"mirror_symmetric": True}})


@pytest.fixture(scope="module", autouse=True)
def _register_fake():
    register_program("fake-molecules", _FakeProgram)


def test_the_engine_resolves_a_template_and_records_the_molecule_it_used(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    solved = QCScanEngine().solve(_system({"template": "ammonium_dimer_cation"}), None)
    recorded = solved.structure["molecule"]
    assert recorded["template"] == "ammonium_dimer_cation" and len(recorded["atoms"]) == 9 and recorded["charge"] == 1
    assert solved.obs("surface_energy").shape == (9, 3)


def test_the_engine_refuses_an_unknown_template_as_a_validation_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValidationError, match="unknown molecule template"):
        QCScanEngine().solve(_system({"template": "nonsense"}), None)


# -- the asymmetric template -----------------------------------------------------------------------------------------------------------------------
def test_the_water_ammonia_ion_is_what_its_parameters_say_and_is_deliberately_asymmetric():
    molecule = water_ammonia_cation()
    pos, sym = positions(molecule), symbols(molecule)
    assert sym == ["O", "N", "H", "H", "H", "H", "H", "H"] and molecule["charge"] == 1
    assert molecule["donor_group"] == [3, 4] and molecule["acceptor_group"] == [5, 6, 7] and sym[0] != sym[1]       # no exchange symmetry is possible
    for h in molecule["donor_group"]:
        bond = pos[h] - pos[0]
        assert np.linalg.norm(bond) == pytest.approx(0.98, abs=1e-12)                                                  # O-H
        assert math.degrees(math.acos(bond[2] / np.linalg.norm(bond))) == pytest.approx(112.0, abs=1e-9)             # from the O...proton axis (+z)
    hoh = math.degrees(math.acos((pos[3] - pos[0]) @ (pos[4] - pos[0]) / 0.98**2))
    assert hoh == pytest.approx(104.5, abs=1e-9)
    for h in molecule["acceptor_group"]:
        bond = pos[h] - pos[1]
        assert np.linalg.norm(bond) == pytest.approx(1.02, abs=1e-12)                                                  # N-H
        assert math.degrees(math.acos(-bond[2] / np.linalg.norm(bond))) == pytest.approx(110.0, abs=1e-9)           # from the N...proton axis (-z)
    azimuth = lambda i: math.degrees(math.atan2(pos[i][1], pos[i][0])) % 360
    separation = min(abs((azimuth(w) - azimuth(n) + 180) % 360 - 180) for w in (3, 4) for n in (5, 6, 7))
    assert sorted(azimuth(h) for h in (5, 6, 7)) == pytest.approx([90.0, 210.0, 330.0], abs=1e-9)                    # the default twist is 90 degrees
    phi = math.degrees(0.5 * math.acos((math.cos(math.radians(104.5)) - math.cos(math.radians(112.0)) ** 2) / math.sin(math.radians(112.0)) ** 2))
    assert separation == pytest.approx(phi - 30.0, abs=1e-9)                                                           # nearest O-H / N-H pair, in azimuth (~28.5)
    turned = positions(water_ammonia_cation(twist=0.0))
    assert np.allclose(turned[:5], pos[:5]) and sorted(math.degrees(math.atan2(turned[h][1], turned[h][0])) % 360 for h in (5, 6, 7)) == pytest.approx([0.0, 120.0, 240.0], abs=1e-9)
    assert HEAVY_ATOM_MASS["water_ammonia_cation"] == pytest.approx(2 * 15.9949 * 14.0031 / (15.9949 + 14.0031))    # twice the O...N reduced mass


class _FakeAsymmetric(QCProgram):
    """An analytic asymmetric surface written out independently of the engine: two Morse diabats with a 1.5 eV offset, coupled."""
    name = "fake-asym"

    def __init__(self):
        self.calls = 0

    def compute(self, jobs):
        out = []
        for job in jobs:
            self.calls += 1
            z_d, z_a, z_h = job.atoms[0][3], job.atoms[1][3], job.atoms[2][3]
            morse = lambda r: 4.6 * (1.0 - math.exp(-2.2 * (r - 0.96))) ** 2
            va, vb = morse(z_h - z_d), morse(z_a - z_h) + 1.5
            ground = 0.5 * (va + vb) - math.sqrt((0.5 * (va - vb)) ** 2 + 0.6**2)
            out.append(QCResult(ground / 27.211386245988, True, 0.1))
        return out


def _asymmetric_system(mirror, template="water_ammonia_cation"):
    system = _system({"template": template})
    system.structure["method"]["program"] = "fake-asym"
    system.structure["scan"] = {"mirror_symmetric": mirror}
    return system


@pytest.mark.parametrize("template", sorted(ASYMMETRIC))
def test_a_false_symmetry_claim_is_refused_for_the_asymmetric_ions_and_the_full_grid_is_computed_without_it(tmp_path, monkeypatch, template):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    fake = _FakeAsymmetric()
    register_program("fake-asym", lambda: fake)
    with pytest.raises(ValidationError, match="not symmetric under exchanging donor and acceptor"):
        QCScanEngine().solve(_asymmetric_system(True, template), None)
    solved = QCScanEngine().solve(_asymmetric_system(False, template), None)
    assert solved.structure["molecule"]["template"] == template
    x, r, e = solved.obs("surface_x"), solved.obs("surface_r"), solved.obs("surface_energy")
    assert e.shape == (9, 3) and solved.obs("calculations_run") + solved.obs("calculations_cached") >= 9 * 3             # no half-grid shortcut
    morse = lambda rr: 4.6 * (1.0 - np.exp(-2.2 * (rr - 0.96))) ** 2
    X, R = np.meshgrid(x, r, indexing="ij")
    va, vb = morse(X + R / 2), morse(R / 2 - X) + 1.5
    expected = 0.5 * (va + vb) - np.sqrt((0.5 * (va - vb)) ** 2 + 0.6**2)
    assert e == pytest.approx(expected - expected.min(), abs=1e-9)                                                        # the formula, at every point
    assert not np.allclose(e, e[::-1], atol=1e-3)                                                                          # and it is not symmetric


def _dihedral(a, b, c, d):
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    return math.degrees(math.atan2(np.cross(n1, n2) @ b2 / np.linalg.norm(b2), n1 @ n2))


def test_the_methanol_water_ion_is_what_its_parameters_say_and_is_mildly_asymmetric():
    molecule = methanol_water_cation()
    pos, sym = positions(molecule), symbols(molecule)
    assert sym == ["O", "O", "H", "H", "H", "H", "C", "H", "H", "H"] and molecule["charge"] == 1
    assert molecule["donor_group"] == [3, 4] and molecule["acceptor_group"] == [5, 6, 7, 8, 9]
    for h in (3, 4):                                                                                       # the water unit is the Zundel one
        bond = pos[h] - pos[0]
        assert np.linalg.norm(bond) == pytest.approx(0.98, abs=1e-12)
        assert math.degrees(math.acos(bond[2] / np.linalg.norm(bond))) == pytest.approx(112.0, abs=1e-9)
    assert math.degrees(math.acos((pos[3] - pos[0]) @ (pos[4] - pos[0]) / 0.98**2)) == pytest.approx(104.5, abs=1e-9)
    angle = lambda a, b, c: math.degrees(math.acos((a - b) @ (c - b) / np.linalg.norm(a - b) / np.linalg.norm(c - b)))
    hydroxyl, carbon, oxygen = pos[5], pos[6], pos[1]
    assert np.linalg.norm(hydroxyl - oxygen) == pytest.approx(0.98, abs=1e-12) and np.linalg.norm(carbon - oxygen) == pytest.approx(1.50, abs=1e-12)
    assert angle(carbon, oxygen, hydroxyl) == pytest.approx(114.0, abs=1e-9)                              # C-O-H
    for bonded in (hydroxyl, carbon):                                                                      # each bond 112 degrees from the O...proton axis (-z)
        bond = bonded - oxygen
        assert math.degrees(math.acos(-bond[2] / np.linalg.norm(bond))) == pytest.approx(112.0, abs=1e-9)
    for h in (7, 8, 9):
        assert np.linalg.norm(pos[h] - carbon) == pytest.approx(1.09, abs=1e-12) and angle(pos[h], carbon, oxygen) == pytest.approx(107.0, abs=1e-9)
    dihedrals = sorted(_dihedral(hydroxyl, oxygen, carbon, pos[h]) for h in (7, 8, 9))
    assert sorted(abs(d) for d in dihedrals) == pytest.approx([60.0, 60.0, 180.0], abs=1e-9)             # staggered: one C-H anti to the O-H
    turned = positions(methanol_water_cation(twist=0.0))
    assert np.allclose(turned[:5], pos[:5]) and not np.allclose(turned[5:], pos[5:])                    # `twist` turns only the methanol unit ...
    assert np.linalg.norm(turned[6] - turned[1]) == pytest.approx(1.50) and turned[6][2] == pytest.approx(pos[6][2])   # ... rigidly about the axis
    assert HEAVY_ATOM_MASS["methanol_water_cation"] == 15.9949                                            # both heavy atoms are oxygens
    scanned = np.array([a[1:] for a in build_geometry(molecule, 0.0, 2.6)])
    distances = np.linalg.norm(scanned[:, None] - scanned[None], axis=-1)
    assert distances[np.triu_indices(10, 1)].min() > 0.97                                                 # nothing closer than a bond


def test_the_methyl_rotor_turns_only_the_methyl_hydrogens_about_the_c_o_bond():
    base, turned = positions(methanol_water_cation()), positions(methanol_water_cation(rotor=30.0))
    assert np.allclose(base[:7], turned[:7]) and not np.allclose(base[7:], turned[7:])             # atoms 0-6 (O, O, H, H, H, hydroxyl H, C) do not move
    carbon, oxygen, hydroxyl = turned[6], turned[1], turned[5]
    dihedrals = sorted(abs(_dihedral(hydroxyl, oxygen, carbon, turned[h])) for h in (7, 8, 9))
    assert dihedrals == pytest.approx([30.0, 90.0, 150.0], abs=1e-9)                                  # anti (180) + 30, then every 120
    for h in (7, 8, 9):
        assert np.linalg.norm(turned[h] - carbon) == pytest.approx(1.09, abs=1e-12)                   # rigid: bond lengths unchanged


def test_the_methanol_and_water_oh_bond_lengths_are_independent_options():
    pos = positions(methanol_water_cation(r_oh=0.95, r_oh_m=1.05))
    assert np.linalg.norm(pos[3] - pos[0]) == pytest.approx(0.95) and np.linalg.norm(pos[4] - pos[0]) == pytest.approx(0.95)    # the water unit
    assert np.linalg.norm(pos[5] - pos[1]) == pytest.approx(1.05)                                                                # the methanol hydroxyl


def test_the_ammonia_methylamine_ion_is_what_its_parameters_say_and_is_mildly_asymmetric():
    molecule = ammonia_methylamine_cation()
    pos, sym = positions(molecule), symbols(molecule)
    assert sym == ["N", "N", "H", "H", "H", "H", "H", "H", "C", "H", "H", "H"] and molecule["charge"] == 1
    assert molecule["donor_group"] == [3, 4, 5] and molecule["acceptor_group"] == [6, 7, 8, 9, 10, 11]
    angle = lambda a, b, c: math.degrees(math.acos((a - b) @ (c - b) / np.linalg.norm(a - b) / np.linalg.norm(c - b)))
    for h in (3, 4, 5):                                                                                    # the ammonia unit is the ammonium dimer's
        bond = pos[h] - pos[0]
        assert np.linalg.norm(bond) == pytest.approx(1.02, abs=1e-12)
        assert math.degrees(math.acos(bond[2] / np.linalg.norm(bond))) == pytest.approx(110.0, abs=1e-9)
    nitrogen, carbon = pos[1], pos[8]
    assert np.linalg.norm(carbon - nitrogen) == pytest.approx(1.50, abs=1e-12)
    for k in (6, 7, 8):                                                                                    # C and both N-H are 110 degrees from the N...proton axis (-z)
        bond = pos[k] - nitrogen
        assert math.degrees(math.acos(-bond[2] / np.linalg.norm(bond))) == pytest.approx(110.0, abs=1e-9)
    for h in (6, 7):
        assert np.linalg.norm(pos[h] - nitrogen) == pytest.approx(1.02, abs=1e-12)
    azimuth = lambda i: math.degrees(math.atan2(pos[i][1], pos[i][0])) % 360
    assert sorted(azimuth(k) for k in (6, 7, 8)) == pytest.approx([60.0, 180.0, 300.0], abs=1e-9)         # three-fold: carbon at the twist, hydrogens 120 apart
    assert azimuth(8) == pytest.approx(60.0, abs=1e-9)
    for h in (9, 10, 11):
        assert np.linalg.norm(pos[h] - carbon) == pytest.approx(1.09, abs=1e-12) and angle(pos[h], carbon, nitrogen) == pytest.approx(108.0, abs=1e-9)
    toward = nitrogen + np.array([0.0, 0.0, -1.0])                                                         # a point towards the shared proton (-z) from the acceptor N
    dihedrals = sorted(abs(_dihedral(toward, nitrogen, carbon, pos[h])) for h in (9, 10, 11))
    assert dihedrals == pytest.approx([60.0, 60.0, 180.0], abs=1e-9)                                      # staggered, one C-H anti to the hydrogen bond
    turned = positions(ammonia_methylamine_cation(twist=0.0))
    assert np.allclose(turned[:6], pos[:6]) and not np.allclose(turned[6:], pos[6:])                    # `twist` turns only the methylamine unit ...
    assert np.linalg.norm(turned[8] - turned[1]) == pytest.approx(1.50) and turned[8][2] == pytest.approx(pos[8][2])    # ... rigidly about the axis
    rotated = positions(ammonia_methylamine_cation(rotor=30.0))
    assert np.allclose(rotated[:9], pos[:9]) and not np.allclose(rotated[9:], pos[9:])                  # `rotor` turns only the methyl hydrogens
    assert sorted(abs(_dihedral(toward, nitrogen, carbon, rotated[h])) for h in (9, 10, 11)) == pytest.approx([30.0, 90.0, 150.0], abs=1e-9)
    assert HEAVY_ATOM_MASS["ammonia_methylamine_cation"] == 14.0031                                       # both heavy atoms are nitrogens
    scanned = np.array([a[1:] for a in build_geometry(molecule, 0.0, 2.7)])
    distances = np.linalg.norm(scanned[:, None] - scanned[None], axis=-1)
    assert distances[np.triu_indices(12, 1)].min() > 1.0                                                  # nothing closer than a bond


def test_the_ammonia_and_methylamine_nh_bond_lengths_are_independent_options():
    pos = positions(ammonia_methylamine_cation(r_nh=0.99, r_nh_m=1.06))
    assert all(np.linalg.norm(pos[h] - pos[0]) == pytest.approx(0.99) for h in (3, 4, 5))                  # the ammonia unit
    assert all(np.linalg.norm(pos[h] - pos[1]) == pytest.approx(1.06) for h in (6, 7))                     # the methylamine N-H


def test_both_methyl_flanks_use_the_same_staggered_group_and_it_is_independent_of_the_flank():
    from substrate.molecules import _methyl_hydrogens
    carbon, heavy = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.5])
    hydrogens = _methyl_hydrogens(carbon, heavy, np.array([1.0, 0.0, 0.3]), 1.09, 109.5, 0.0)
    assert all(np.linalg.norm(h - carbon) == pytest.approx(1.09) for h in hydrogens)
    assert all(math.degrees(math.acos(h[2] / np.linalg.norm(h - carbon))) == pytest.approx(109.5) for h in hydrogens)       # from the C-heavy axis (+z)
    first = hydrogens[0] - carbon
    assert first[0] < 0 and first[1] == pytest.approx(0.0, abs=1e-12)                                      # the first is anti to the reference direction (+x)
    assert sorted(math.degrees(math.atan2(h[1], h[0])) % 360 for h in hydrogens) == pytest.approx([60.0, 180.0, 300.0], abs=1e-9)           # 120 degrees apart


def test_the_ammonia_and_methylamine_polar_angles_are_independent_options():
    pos = positions(ammonia_methylamine_cation(theta_a=105.0, theta_m=115.0))
    for h in (3, 4, 5):
        bond = pos[h] - pos[0]
        assert math.degrees(math.acos(bond[2] / np.linalg.norm(bond))) == pytest.approx(105.0, abs=1e-9)    # from the N...proton axis (+z) at the donor
    for k in (6, 7, 8):
        bond = pos[k] - pos[1]
        assert math.degrees(math.acos(-bond[2] / np.linalg.norm(bond))) == pytest.approx(115.0, abs=1e-9)   # and from the axis (-z) at the acceptor


def test_the_fluoride_methanol_anion_is_what_its_parameters_say_and_is_mildly_asymmetric():
    molecule = fluoride_methanol_anion()
    pos, sym = positions(molecule), symbols(molecule)
    assert sym == ["F", "O", "H", "C", "H", "H", "H"] and molecule["charge"] == -1
    assert molecule["donor_group"] == [] and molecule["acceptor_group"] == [3, 4, 5, 6]                    # a bare fluorine, and a methoxy group on the oxygen
    angle = lambda a, b, c: math.degrees(math.acos((a - b) @ (c - b) / np.linalg.norm(a - b) / np.linalg.norm(c - b)))
    oxygen, carbon = pos[1], pos[3]
    assert np.linalg.norm(carbon - oxygen) == pytest.approx(1.43, abs=1e-12)
    assert math.degrees(math.acos(-(carbon - oxygen)[2] / np.linalg.norm(carbon - oxygen))) == pytest.approx(109.0, abs=1e-9)   # from the O...proton axis (-z)
    for h in (4, 5, 6):
        assert np.linalg.norm(pos[h] - carbon) == pytest.approx(1.09, abs=1e-12) and angle(pos[h], carbon, oxygen) == pytest.approx(108.0, abs=1e-9)
    toward = oxygen + np.array([0.0, 0.0, -1.0])                                                           # a point towards the shared proton from the oxygen
    assert sorted(abs(_dihedral(toward, oxygen, carbon, pos[h])) for h in (4, 5, 6)) == pytest.approx([60.0, 60.0, 180.0], abs=1e-9)   # staggered, one anti
    turned = positions(fluoride_methanol_anion(twist=90.0))
    assert np.allclose(turned[:3], pos[:3]) and not np.allclose(turned[3:], pos[3:])                    # `twist` turns only the methoxy group ...
    assert np.linalg.norm(turned[3] - turned[1]) == pytest.approx(1.43) and turned[3][2] == pytest.approx(pos[3][2])          # ... rigidly about the axis
    azimuth = lambda i, coordinates: math.degrees(math.atan2(coordinates[i][1], coordinates[i][0])) % 360
    assert azimuth(3, pos) == pytest.approx(0.0, abs=1e-9) and azimuth(3, turned) == pytest.approx(90.0, abs=1e-9)           # the angle is in degrees
    rotated = positions(fluoride_methanol_anion(rotor=30.0))
    assert np.allclose(rotated[:4], pos[:4]) and not np.allclose(rotated[4:], pos[4:])                  # `rotor` turns only the methyl hydrogens
    assert sorted(abs(_dihedral(toward, oxygen, carbon, rotated[h])) for h in (4, 5, 6)) == pytest.approx([30.0, 90.0, 150.0], abs=1e-9)
    assert HEAVY_ATOM_MASS["fluoride_methanol_anion"] == pytest.approx(2 * 15.9949 * 18.9984 / (15.9949 + 18.9984))       # twice the O...F reduced mass
    scanned = np.array([a[1:] for a in build_geometry(molecule, 0.0, 2.6)])
    distances = np.linalg.norm(scanned[:, None] - scanned[None], axis=-1)
    assert distances[np.triu_indices(7, 1)].min() > 1.0                                                   # nothing closer than a bond
    assert exchange_operations(molecule) == ([], [])


def test_the_methoxy_bond_length_and_angle_are_independent_options():
    pos = positions(fluoride_methanol_anion(r_oc=1.50, theta_c=115.0))
    bond = pos[3] - pos[1]
    assert np.linalg.norm(bond) == pytest.approx(1.50) and math.degrees(math.acos(-bond[2] / np.linalg.norm(bond))) == pytest.approx(115.0, abs=1e-9)
    assert np.allclose(pos[:3], positions(fluoride_methanol_anion())[:3])                                  # the heavy atoms and the proton do not move


def test_the_donor_side_of_the_fluoride_methanol_ion_is_a_bare_fluorine_like_the_bifluoride():
    assert fluoride_methanol_anion()["donor_group"] == bifluoride_anion()["donor_group"] == []
    assert fluoride_methanol_anion()["atoms"][0][0] == bifluoride_anion()["atoms"][0][0] == "F"


def test_the_chloride_hf_ion_is_a_bare_collinear_triatomic_that_differs_from_the_bifluoride_only_in_one_heavy_atom():
    molecule, parent = chloride_hf_anion(), bifluoride_anion()
    pos, sym = positions(molecule), symbols(molecule)
    assert sym == ["Cl", "F", "H"] and molecule["charge"] == -1 and np.allclose(pos[:, :2], 0.0)             # all on the axis, collinear
    assert molecule["donor_group"] == molecule["acceptor_group"] == [] and molecule["proton"] == 2             # nothing flanks either end
    assert (molecule["donor"], molecule["acceptor"]) == (parent["donor"], parent["acceptor"]) and symbols(parent) == ["F", "F", "H"]
    assert pos[0][2] < pos[2][2] < pos[1][2]                                                                   # chloride below, fluoride above, the proton between
    assert HEAVY_ATOM_MASS["chloride_hf_anion"] == pytest.approx(2 * 34.96885 * 18.9984 / (34.96885 + 18.9984))   # twice the Cl...F reduced mass
    assert exchange_operations(molecule) == ([], [])                                                           # no exchange symmetry: two different heavy atoms
    scanned = build_geometry(molecule, 0.3, 2.9)
    assert [a[3] for a in scanned] == pytest.approx([-1.45, 1.45, 0.3])                                         # heavy atoms at -+R/2, the proton at x
    assert [a[0] for a in scanned] == ["Cl", "F", "H"]
