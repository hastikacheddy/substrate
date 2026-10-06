"""Molecule templates for the quantum-chemistry engine (`electronic.qc_scan_2d`).

Each function returns the engine's `molecule` structure: reference atoms (angstrom) with the donor, proton and acceptor on
the z axis, and the atoms that travel with each heavy atom listed as its group.

An experiment file may name a template instead of listing atoms: `molecule: {template: zundel_cation}` (with optional `options`),
which keeps files short and is resolved, recorded in full on the solved system, by `resolve_molecule`.
"""
from __future__ import annotations

import math

import numpy as np


def _water_flank(z_oxygen: float, towards_proton: int, twist: float, r_oh: float, theta_c: float, hoh: float) -> list[list]:
    """Two hydrogens of a pyramidal H3O+-like unit on an oxygen at z_oxygen; the third (shared) proton lies on the +z axis
    if towards_proton = +1, on the -z axis if -1. `twist` rotates the unit about z (radians).

    theta_c is the angle between each O-H and the O...shared-proton axis; hoh the H-O-H angle of the two flank hydrogens."""
    th = math.radians(theta_c)
    cos_two_phi = (math.cos(math.radians(hoh)) - math.cos(th) ** 2) / math.sin(th) ** 2
    phi = 0.5 * math.acos(cos_two_phi)
    atoms = []
    for sign in (+1, -1):
        azimuth = sign * phi + twist
        atoms.append([
            "H",
            r_oh * math.sin(th) * math.cos(azimuth),
            r_oh * math.sin(th) * math.sin(azimuth),
            z_oxygen + towards_proton * r_oh * math.cos(th),
        ])
    return atoms


def zundel_cation(r_oh: float = 0.98, theta_c: float = 112.0, hoh: float = 104.5) -> dict:
    """H5O2+: two water-like units sharing a proton, staggered (the two H-O-H planes perpendicular).

    The flank geometry is frozen at an H3O+-like shape (O-H = 0.98 angstrom, 112 degrees between each O-H and the shared
    proton axis, H-O-H = 104.5 degrees). It is symmetric under exchanging donor and acceptor (a two-fold axis across the
    O-O line at 45 degrees to the two H-O-H planes), so `mirror_symmetric` may be used with it. Atom order: 0 donor O, 1 acceptor O, 2 shared proton, 3-4 donor hydrogens, 5-6 acceptor hydrogens.
    """
    atoms = [["O", 0.0, 0.0, -1.2], ["O", 0.0, 0.0, 1.2], ["H", 0.0, 0.0, 0.0]]
    atoms += _water_flank(-1.2, +1, 0.0, r_oh, theta_c, hoh)
    atoms += _water_flank(+1.2, -1, math.pi / 2.0, r_oh, theta_c, hoh)
    return {
        "atoms": atoms, "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [3, 4], "acceptor_group": [5, 6],
        "charge": 1, "spin": 0,
    }


def _ammonia_flank(z_nitrogen: float, towards_proton: int, twist: float, r_nh: float, theta_c: float) -> list[list]:
    """Three hydrogens of an NH3-like unit on a nitrogen; the shared proton lies on the +z axis if towards_proton = +1.
    theta_c is the angle between each N-H and the N...shared-proton axis. `twist` rotates the unit about z (radians)."""
    th = math.radians(theta_c)
    return [
        ["H", r_nh * math.sin(th) * math.cos(2.0 * math.pi * k / 3.0 + twist),
         r_nh * math.sin(th) * math.sin(2.0 * math.pi * k / 3.0 + twist),
         z_nitrogen + towards_proton * r_nh * math.cos(th)]
        for k in range(3)
    ]


def ammonium_dimer_cation(r_nh: float = 1.02, theta_c: float = 110.0) -> dict:
    """N2H7+: two ammonia-like units sharing a proton (an N-H-N hydrogen bond), staggered so the ion has inversion symmetry.

    Flank geometry is frozen at an NH4+-like shape (N-H = 1.02 angstrom, 110 degrees between each N-H and the shared-proton
    axis). Symmetric under exchanging donor and acceptor. Atom order: 0 donor N, 1 acceptor N, 2 shared proton, 3-5 donor
    hydrogens, 6-8 acceptor hydrogens."""
    atoms = [["N", 0.0, 0.0, -1.3], ["N", 0.0, 0.0, 1.3], ["H", 0.0, 0.0, 0.0]]
    atoms += _ammonia_flank(-1.3, +1, 0.0, r_nh, theta_c)
    atoms += _ammonia_flank(+1.3, -1, math.pi / 3.0, r_nh, theta_c)
    return {
        "atoms": atoms, "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [3, 4, 5], "acceptor_group": [6, 7, 8],
        "charge": 1, "spin": 0,
    }


def bifluoride_anion() -> dict:
    """FHF-: a proton between two fluorides. A bare collinear triatomic: no flanking groups, so the rigid-group approximation
    is exact here. Symmetric. Atom order: 0 donor F, 1 acceptor F, 2 shared proton."""
    return {
        "atoms": [["F", 0.0, 0.0, -1.15], ["F", 0.0, 0.0, 1.15], ["H", 0.0, 0.0, 0.0]],
        "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [], "acceptor_group": [],
        "charge": -1, "spin": 0,
    }


def water_ammonia_cation(r_oh: float = 0.98, theta_o: float = 112.0, hoh: float = 104.5, r_nh: float = 1.02, theta_n: float = 110.0,
                         twist: float = 90.0) -> dict:
    """[H2O...H...NH3]+: a proton between a water and an ammonia, the first ASYMMETRIC template. Ammonia holds a proton far more tightly
    than water (the gas-phase proton affinities differ by about 1.7 eV), so the proton sits on the nitrogen and the surface is not
    symmetric under exchanging donor and acceptor: do NOT use `mirror_symmetric` with it (the engine would refuse).

    Donor = the water oxygen, acceptor = the ammonia nitrogen, so the reactant (proton on oxygen) is the high-energy side. The flank
    geometry is frozen at an H3O+-like water unit (O-H = 0.98 angstrom, 112 degrees between each O-H and the O...proton axis,
    H-O-H = 104.5 degrees) and an NH4+-like ammonia unit (N-H = 1.02 angstrom, 110 degrees). `twist` (degrees) turns the ammonia unit about
    the axis relative to the water unit: the water hydrogens lie at azimuth +-58.4 degrees, the ammonia ones at twist, twist + 120 and
    twist + 240, so at the default 90 degrees the nearest pair is about 28 degrees apart (0 would give 58 degrees, the most staggered). The choice is
    arbitrary and, being 2.6 angstrom apart along the axis, matters little (see the README for how little). Atom order: 0 donor O, 1 acceptor N,
    2 shared proton, 3-4 water hydrogens, 5-7 ammonia hydrogens."""
    atoms = [["O", 0.0, 0.0, -1.3], ["N", 0.0, 0.0, 1.3], ["H", 0.0, 0.0, 0.0]]
    atoms += _water_flank(-1.3, +1, 0.0, r_oh, theta_o, hoh)
    atoms += _ammonia_flank(+1.3, -1, math.radians(twist), r_nh, theta_n)
    return {
        "atoms": atoms, "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [3, 4], "acceptor_group": [5, 6, 7],
        "charge": 1, "spin": 0,
    }


def _methanol_flank(z_oxygen: float, towards_proton: int, twist: float, r_oh: float, r_oc: float, theta_c: float, coh: float,
                    r_ch: float, hco: float, rotor: float = 0.0) -> list[list]:
    """The substituents of a protonated-methanol oxygen at z_oxygen: its hydroxyl hydrogen, its carbon and the carbon's three hydrogens,
    in that order. The shared proton lies on the +z axis if towards_proton = +1, on the -z axis if -1.

    theta_c is the angle between each O-substituent bond and the O...shared-proton axis, coh the C-O-H angle; the hydroxyl hydrogen sits at
    azimuth +phi and the carbon at -phi (plus `twist`), where phi follows from coh. The methyl hydrogens are staggered about the C-O axis with
    the first one anti to the hydroxyl hydrogen (H-O-C-H dihedral 180 degrees) unless `rotor` (degrees) turns the methyl group away from that,
    r_ch long and hco from the C-O axis."""
    th = math.radians(theta_c)
    phi = 0.5 * math.acos((math.cos(math.radians(coh)) - math.cos(th) ** 2) / math.sin(th) ** 2)

    def substituent(length: float, azimuth: float) -> np.ndarray:
        return np.array([length * math.sin(th) * math.cos(azimuth), length * math.sin(th) * math.sin(azimuth),
                         z_oxygen + towards_proton * length * math.cos(th)])

    oxygen = np.array([0.0, 0.0, z_oxygen])
    hydroxyl, carbon = substituent(r_oh, phi + twist), substituent(r_oc, -phi + twist)
    methyl = _methyl_hydrogens(carbon, oxygen, hydroxyl - oxygen, r_ch, hco, rotor)          # one C-H anti to the hydroxyl O-H
    return [["H", *map(float, hydroxyl)], ["C", *map(float, carbon)]] + [["H", *map(float, h)] for h in methyl]


def _methyl_hydrogens(carbon: np.ndarray, heavy: np.ndarray, anti_to: np.ndarray, r_ch: float, hco: float, rotor: float) -> list[np.ndarray]:
    """The three hydrogens of a methyl group on `carbon`, bonded to the heavy atom `heavy`: each r_ch from the carbon and hco degrees from the
    C-heavy axis, 120 degrees apart about it, staggered so that the first is anti to the direction `anti_to` (seen from the heavy atom) unless
    `rotor` (degrees) turns the group about the C-heavy bond."""
    u = (heavy - carbon) / np.linalg.norm(heavy - carbon)                       # from the carbon towards the heavy atom
    w = anti_to - (anti_to @ u) * u                                              # that direction, perpendicular to the C-heavy axis
    e1 = -w / np.linalg.norm(w)                                                  # pointing away from it: anti
    e2 = np.cross(u, e1)
    hco_rad = math.radians(hco)
    return [carbon + r_ch * (math.cos(hco_rad) * u + math.sin(hco_rad) * (math.cos(b) * e1 + math.sin(b) * e2))
            for b in (math.radians(rotor), math.radians(rotor) + 2.0 * math.pi / 3.0, math.radians(rotor) + 4.0 * math.pi / 3.0)]


def _methylamine_flank(z_nitrogen: float, towards_proton: int, twist: float, r_nh: float, r_nc: float, theta_n: float, r_ch: float,
                       hcn: float, rotor: float = 0.0) -> list[list]:
    """The substituents of a protonated-methylamine nitrogen at z_nitrogen: two N-H hydrogens, the carbon, and the carbon's three hydrogens,
    in that order. The shared proton lies on the +z axis if towards_proton = +1, on the -z axis if -1.

    The carbon and the two hydrogens sit theta_n degrees from the N...shared-proton axis, 120 degrees apart in azimuth, the carbon at
    azimuth `twist` and the hydrogens at twist + 120 and twist + 240. The methyl hydrogens are staggered about the C-N axis with the first
    anti to the shared proton (P-N-C-H dihedral 180 degrees) unless `rotor` (degrees) turns the group, r_ch long and hcn from the C-N axis."""
    th = math.radians(theta_n)

    def substituent(length: float, azimuth: float) -> np.ndarray:
        return np.array([length * math.sin(th) * math.cos(azimuth), length * math.sin(th) * math.sin(azimuth),
                         z_nitrogen + towards_proton * length * math.cos(th)])

    nitrogen = np.array([0.0, 0.0, z_nitrogen])
    carbon = substituent(r_nc, twist)
    hydrogens = [substituent(r_nh, twist + 2.0 * math.pi / 3.0), substituent(r_nh, twist + 4.0 * math.pi / 3.0)]
    proton_direction = np.array([0.0, 0.0, float(towards_proton)])
    methyl = _methyl_hydrogens(carbon, nitrogen, proton_direction, r_ch, hcn, rotor)         # one C-H anti to the hydrogen bond
    return [["H", *map(float, h)] for h in hydrogens] + [["C", *map(float, carbon)]] + [["H", *map(float, h)] for h in methyl]


def ammonia_methylamine_cation(r_nh: float = 1.02, theta_a: float = 110.0, r_nh_m: float = 1.02, r_nc: float = 1.50, theta_m: float = 110.0,
                               r_ch: float = 1.09, hcn: float = 108.0, twist: float = 60.0, rotor: float = 0.0) -> dict:
    """[H3N...H...H2NCH3]+: a proton between an ammonia and a methylamine, a MILDLY asymmetric N-H-N ion: the nitrogen counterpart of the
    methanol-water ion, and the one-methyl derivative of the symmetric ammonium dimer (methylamine holds a proton about 0.47 eV more tightly
    than ammonia, in the gas phase). The proton prefers the methylamine nitrogen; do NOT use `mirror_symmetric` with it.

    Donor = the ammonia nitrogen (the lower-proton-affinity side, so the reactant, proton on ammonia, is the higher-energy well), acceptor =
    the methylamine nitrogen. The donor flank is the NH4+-like ammonia unit of the ammonium-dimer template (N-H `r_nh`, `theta_a` degrees from
    the N...proton axis); the acceptor flank is protonated methylamine: N-H `r_nh_m`, N-C `r_nc`, each `theta_m` degrees from the axis, a methyl
    with C-H `r_ch` and H-C-N = `hcn`, staggered with one C-H anti to the hydrogen bond. `twist` turns the methylamine unit about the axis
    relative to the ammonia unit (60 degrees staggers them), `rotor` turns its methyl group about the C-N bond. All of this is frozen during a
    scan. Atom order: 0 donor N, 1 acceptor N, 2 shared proton, 3-5 ammonia hydrogens, 6-7 methylamine N-H hydrogens, 8 carbon, 9-11 methyl hydrogens."""
    atoms = [["N", 0.0, 0.0, -1.3], ["N", 0.0, 0.0, 1.3], ["H", 0.0, 0.0, 0.0]]
    atoms += _ammonia_flank(-1.3, +1, 0.0, r_nh, theta_a)
    atoms += _methylamine_flank(+1.3, -1, math.radians(twist), r_nh_m, r_nc, theta_m, r_ch, hcn, rotor)
    return {
        "atoms": atoms, "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [3, 4, 5], "acceptor_group": [6, 7, 8, 9, 10, 11],
        "charge": 1, "spin": 0,
    }


def _methoxy_flank(z_oxygen: float, towards_proton: int, twist: float, r_oc: float, theta_c: float, r_ch: float, hco: float,
                   rotor: float = 0.0) -> list[list]:
    """The substituents of a methoxy / methanol oxygen at z_oxygen: the carbon and its three hydrogens, in that order. The shared proton
    lies on the +z axis if towards_proton = +1, on the -z axis if -1.

    The carbon is r_oc from the oxygen and theta_c degrees from the O...shared-proton axis, at azimuth `twist` (an oxygen with a single
    substituent has no other azimuth to be measured against, so `twist` only turns the whole group about the axis). The methyl hydrogens
    are staggered about the C-O axis with the first anti to the shared proton (P-O-C-H dihedral 180 degrees) unless `rotor` (degrees)
    turns the group, r_ch long and hco from the C-O axis."""
    th = math.radians(theta_c)
    oxygen = np.array([0.0, 0.0, z_oxygen])
    carbon = np.array([r_oc * math.sin(th) * math.cos(twist), r_oc * math.sin(th) * math.sin(twist),
                       z_oxygen + towards_proton * r_oc * math.cos(th)])
    methyl = _methyl_hydrogens(carbon, oxygen, np.array([0.0, 0.0, float(towards_proton)]), r_ch, hco, rotor)   # one C-H anti to the hydrogen bond
    return [["C", *map(float, carbon)]] + [["H", *map(float, h)] for h in methyl]


def fluoride_methanol_anion(r_oc: float = 1.43, theta_c: float = 109.0, r_ch: float = 1.09, hco: float = 108.0, twist: float = 0.0,
                            rotor: float = 0.0) -> dict:
    """[F...H...OCH3]-: a proton between a fluoride and a methoxide, the fluoride-methanol complex (or, with the proton on the fluorine,
    methoxide-hydrogen fluoride): a MILDLY asymmetric F-H-O ANION, the nearest asymmetric neighbour of the symmetric bifluoride ion that
    carries a methyl group. It is NOT a pure one-substituent derivative of FHF-, which has no heavy-atom hydrogen to replace: one fluoride
    is swapped for methoxide, so the heavy atom changes as well as the methyl appearing.

    Methoxide is the stronger base (gas-phase acidities, hydrogen fluoride ~1554 and methanol ~1590 kJ/mol, each uncertain by ~8; a gap of
    about 0.4 eV), so the proton prefers the oxygen; do NOT use `mirror_symmetric` with it. Donor = the fluorine (the lower-proton-affinity
    side, so the reactant, proton on fluorine, is the higher-energy well), acceptor = the methoxide oxygen. The donor flank is empty (a bare
    fluorine, as in the bifluoride template); the acceptor flank is a methoxy group: C-O `r_oc`, `theta_c` degrees from the O...proton axis,
    a methyl with C-H `r_ch` and H-C-O = `hco`, staggered with one C-H anti to the hydrogen bond (`rotor` turns it, `twist` turns the whole
    group about the axis). Frozen during a scan. Atom order: 0 donor F, 1 acceptor O, 2 shared proton, 3 carbon, 4-6 methyl hydrogens."""
    atoms = [["F", 0.0, 0.0, -1.25], ["O", 0.0, 0.0, 1.25], ["H", 0.0, 0.0, 0.0]]
    atoms += _methoxy_flank(+1.25, -1, math.radians(twist), r_oc, theta_c, r_ch, hco, rotor)
    return {
        "atoms": atoms, "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [], "acceptor_group": [3, 4, 5, 6],
        "charge": -1, "spin": 0,
    }


def chloride_hf_anion() -> dict:
    """[Cl...H...F]-: a proton between a chloride and a fluoride, the chloride-hydrogen fluoride complex Cl-...HF: a STRONGLY asymmetric
    F-H-Cl ANION, the strong counterpart of the mildly asymmetric fluoride-methanol ion and, like it, an asymmetric neighbour of the symmetric
    bifluoride ion (a fluoride swapped for chloride). Fluoride is by far the stronger base (gas-phase acidities, hydrogen fluoride ~1554 and
    hydrogen chloride ~1395 kJ/mol, a gap of about 1.65 eV, like water-ammonia's 1.7), so the proton sits on the fluorine; do NOT use
    `mirror_symmetric` with it. Donor = the chloride (the lower-proton-affinity side, so the reactant, proton on chlorine, is the higher-
    energy well), acceptor = the fluoride. Like the bifluoride, a bare collinear triatomic: no flanking groups, so the rigid-group
    approximation is exact. Atom order: 0 donor Cl, 1 acceptor F, 2 shared proton."""
    return {
        "atoms": [["Cl", 0.0, 0.0, -1.45], ["F", 0.0, 0.0, 1.45], ["H", 0.0, 0.0, 0.0]],
        "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [], "acceptor_group": [],
        "charge": -1, "spin": 0,
    }


def methanol_water_cation(r_oh: float = 0.98, theta_o: float = 112.0, hoh: float = 104.5, r_oh_m: float = 0.98, r_oc: float = 1.50,
                          theta_m: float = 112.0, coh: float = 114.0, r_ch: float = 1.09, hco: float = 107.0, twist: float = 90.0, rotor: float = 0.0) -> dict:
    """[H2O...H...HOCH3]+: a proton between a water and a methanol, a MILDLY asymmetric O-H-O ion (the gas-phase proton affinities of methanol
    and water differ by about 0.65 eV, against about 1.7 eV for ammonia and water): between the symmetric Zundel ion and the strongly
    asymmetric water-ammonia ion. The proton prefers the methanol oxygen; do NOT use `mirror_symmetric` with it.

    Donor = the water oxygen (the lower-proton-affinity side, so the reactant, proton on water, is the higher-energy well), acceptor = the
    methanol oxygen. The donor flank is the H3O+-like water unit of the Zundel template; the acceptor flank is protonated methanol: O-H
    `r_oh_m`, O-C `r_oc` (angstrom), each at `theta_m` degrees from the O...proton axis with C-O-H = `coh`, a methyl group with C-H `r_ch` and
    H-C-O = `hco`, staggered with one C-H anti to the O-H. `twist` turns the methanol unit about the axis relative to the water unit, `rotor` turns its methyl group about the C-O bond. All of this
    is frozen during a scan. Atom order: 0 donor O, 1 acceptor O, 2 shared proton, 3-4 water hydrogens, 5 hydroxyl hydrogen, 6 carbon,
    7-9 methyl hydrogens."""
    atoms = [["O", 0.0, 0.0, -1.3], ["O", 0.0, 0.0, 1.3], ["H", 0.0, 0.0, 0.0]]
    atoms += _water_flank(-1.3, +1, 0.0, r_oh, theta_o, hoh)
    atoms += _methanol_flank(+1.3, -1, math.radians(twist), r_oh_m, r_oc, theta_m, coh, r_ch, hco, rotor)
    return {
        "atoms": atoms, "donor": 0, "acceptor": 1, "proton": 2, "donor_group": [3, 4], "acceptor_group": [5, 6, 7, 8, 9],
        "charge": 1, "spin": 0,
    }


#: heavy-atom masses (amu) for the templates, for the molecular and quantum scales. The two heavy atoms of the asymmetric template differ;
#: the model engines take one mass and use half of it as the reduced mass of the heavy-atom coordinate, so the entry is twice the O...N
#: reduced mass: 2 * (15.9949 * 14.0031) / (15.9949 + 14.0031).
HEAVY_ATOM_MASS = {"zundel_cation": 15.9949, "ammonium_dimer_cation": 14.0031, "bifluoride_anion": 18.9984,
                   "water_ammonia_cation": 2.0 * (15.9949 * 14.0031) / (15.9949 + 14.0031), "methanol_water_cation": 15.9949, "ammonia_methylamine_cation": 14.0031,
                   "fluoride_methanol_anion": 2.0 * (15.9949 * 18.9984) / (15.9949 + 18.9984),
                   "chloride_hf_anion": 2.0 * (34.96885 * 18.9984) / (34.96885 + 18.9984)}

TEMPLATES = {"zundel_cation": zundel_cation, "ammonium_dimer_cation": ammonium_dimer_cation, "bifluoride_anion": bifluoride_anion,
             "water_ammonia_cation": water_ammonia_cation, "methanol_water_cation": methanol_water_cation,
             "ammonia_methylamine_cation": ammonia_methylamine_cation,
             "fluoride_methanol_anion": fluoride_methanol_anion,
             "chloride_hf_anion": chloride_hf_anion}


def resolve_molecule(spec: dict) -> dict:
    """A molecule spec that names a template (`{template: name, options: {...}}`) becomes the full structure; a spec that lists
    atoms is returned unchanged. Explicit keys next to `template` (e.g. `charge`) override the template's. The result keeps the
    template's name under `template`, so a solved system records which molecule it was; resolving a resolved spec is harmless."""
    if "template" not in spec:
        return spec
    name = spec["template"]
    if name not in TEMPLATES:
        raise ValueError(f"unknown molecule template '{name}' (have: {', '.join(TEMPLATES)})")
    resolved = TEMPLATES[name](**spec.get("options", {}))
    resolved.update({k: v for k, v in spec.items() if k != "options"})
    return resolved
