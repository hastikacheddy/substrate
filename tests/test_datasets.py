"""Reference values from published datasets: the transcription, the unit handling, and the physics the numbers must obey."""
import math

import pytest

from substrate import ValidationError
from substrate.datasets import (
    KJ_PER_MOL_PER_EV, TEMPLATE_BASES, ReferenceValue, experimental_well_gap, load_reference_set,
)
from substrate.molecules import TEMPLATES

NIST = load_reference_set()


def test_a_kilojoule_per_mole_is_converted_with_the_exact_avogadro_times_charge():
    """N_A e = 96485.33212... C/mol, so 1 eV is 96.485... kJ/mol."""
    assert KJ_PER_MOL_PER_EV == pytest.approx(6.02214076e23 * 1.602176634e-19 / 1000.0, rel=1e-12)
    one = ReferenceValue("x", "proton_affinity", "H2O", KJ_PER_MOL_PER_EV, 2 * KJ_PER_MOL_PER_EV, "m", "r").quantity_ev()
    assert (one.value, one.sigma, one.unit) == (pytest.approx(1.0), pytest.approx(2.0), "eV")
    value = ReferenceValue("x", "proton_affinity", "H2O", 96.0, None, "m", "the citation")
    assert value.quantity_ev().source == "the citation" and value.quantity_ev("elsewhere").source == "elsewhere" and value.quantity_ev().sigma is None


def test_the_nist_set_records_where_it_came_from():
    assert NIST.source["database"].startswith("NIST Chemistry WebBook") and NIST.source["retrieved"] == "2026-10-08" and NIST.source["unit"] == "kJ/mol"
    assert set(NIST.species) == {"H2O", "NH3", "CH3OH", "CH3NH2", "HF", "HCl"}
    assert all(spec["cas"].startswith("C") and spec["basic_atom"] in ("O", "N", "F", "Cl") for spec in NIST.species.values())
    assert all(v.reference and v.method for v in NIST.values.values())                        # every entry names its source and method
    q = NIST.quantity("pa_nh3")
    assert q.unit == "eV" and q.value == pytest.approx(853.6 / KJ_PER_MOL_PER_EV) and "Hunter and Lias, 1998" in q.source and "2026-10-08" in q.source


def test_proton_affinities_follow_the_chemistry_they_encode():
    """Independent of the transcription: a methyl group makes the amine and the alcohol the stronger base, ammonia beats water by far, and all
    four sit in the range of small neutral bases."""
    pa = {k: NIST.get(f"pa_{k}").value for k in ("h2o", "nh3", "ch3oh", "ch3nh2")}
    assert pa["ch3nh2"] > pa["nh3"] > pa["ch3oh"] > pa["h2o"]
    assert pa["ch3oh"] > pa["h2o"] and pa["ch3nh2"] > pa["nh3"]                                # the methyl effect, in both families
    assert 600 < min(pa.values()) and max(pa.values()) < 1000                                  # kJ/mol


def test_the_gibbs_acidities_are_below_the_enthalpies_by_a_fraction_of_the_free_protons_entropy_term():
    """Delta G = Delta H - T Delta S, and dissociating into a free proton gains entropy (the proton alone has T S = 32 kJ/mol at 298 K), so each
    Gibbs acidity is 15-40 kJ/mol below its enthalpy: a check on the transcribed numbers that does not use the numbers themselves."""
    for species in ("hf", "hcl", "ch3oh"):
        difference = NIST.get(f"acidity_{species}").value - NIST.get(f"acidity_gibbs_{species}").value
        assert 15.0 < difference < 40.0, species


def test_acidities_order_the_anions_as_proton_affinities_order_the_bases():
    """The anion that holds the proton more tightly belongs to the weaker acid: methoxide, then fluoride, then chloride."""
    acidity = {k: NIST.get(f"acidity_{k}").value for k in ("ch3oh", "hf", "hcl")}
    assert acidity["ch3oh"] > acidity["hf"] > acidity["hcl"]


def test_the_gap_between_two_bases_is_their_difference_in_ev_with_the_uncertainties_in_quadrature():
    water_ammonia = NIST.gap("pa_h2o", "pa_nh3")
    assert water_ammonia.value == pytest.approx((853.6 - 691.0) / KJ_PER_MOL_PER_EV) and water_ammonia.value == pytest.approx(1.685, abs=1e-3)
    assert water_ammonia.sigma is None                                                      # NIST lists no uncertainty for the proton affinities: none is invented
    fluoride_methoxide = NIST.gap("acidity_hf", "acidity_ch3oh")
    assert fluoride_methoxide.value == pytest.approx(42.0 / KJ_PER_MOL_PER_EV)
    assert fluoride_methoxide.sigma == pytest.approx(math.hypot(5.0, 8.0) / KJ_PER_MOL_PER_EV)
    same = NIST.gap("pa_h2o", "pa_h2o")
    assert (same.value, same.sigma) == (0.0, 0.0)
    assert "Hunter and Lias" in water_ammonia.source and "retrieved" in water_ammonia.source
    with pytest.raises(ValidationError, match="cannot take the gap between a proton_affinity and a acidity"):
        NIST.gap("pa_h2o", "acidity_hf")
    with pytest.raises(ValidationError, match="no reference value 'nope'"):
        NIST.get("nope")


def test_a_gap_has_an_uncertainty_only_when_both_entries_list_one():
    from substrate.datasets import ReferenceSet
    a = ReferenceValue("a", "acidity", "X", 1000.0, 3.0, "m", "r")
    b = ReferenceValue("b", "acidity", "X", 1100.0, None, "m", "r")
    c = ReferenceValue("c", "acidity", "X", 1200.0, 4.0, "m", "r")
    reference_set = ReferenceSet({"database": "d", "retrieved": "t"}, {"X": {}}, {"a": a, "b": b, "c": c})
    assert reference_set.gap("a", "b").sigma is None and reference_set.gap("b", "c").sigma is None        # one missing: none is made up
    assert reference_set.gap("a", "c").sigma == pytest.approx(5.0 / KJ_PER_MOL_PER_EV)                  # 3 and 4 in quadrature


def test_every_template_has_a_measured_gap_which_is_zero_only_for_the_symmetric_ones_and_never_negative():
    assert set(TEMPLATE_BASES) == set(TEMPLATES)                                            # a new template must say which bases it joins
    symmetric = {"zundel_cation", "ammonium_dimer_cation", "bifluoride_anion"}
    for template in TEMPLATES:
        gap = experimental_well_gap(template)
        assert gap.unit == "eV" and gap.value >= 0.0, template
        assert (gap.value == 0.0) == (template in symmetric), template
    expected = {"water_ammonia_cation": 1.685, "methanol_water_cation": 0.656, "ammonia_methylamine_cation": 0.470,
                "fluoride_methanol_anion": 0.435, "chloride_hf_anion": 1.659}                   # eV, from the kJ/mol differences 162.6, 63.3, 45.4, 42, 160.1
    for template, ev in expected.items():
        assert experimental_well_gap(template).value == pytest.approx(ev, abs=1e-3), template
    with pytest.raises(ValidationError, match="no measured proton-affinity pair"):
        experimental_well_gap("unknown")


def test_the_donor_side_is_the_weaker_base_and_binds_through_the_templates_donor_atom():
    """The data and the geometry must agree about which side is which: the weaker base's proton-binding element is the template's donor atom
    and the stronger base's is its acceptor atom (the conjugate base of HF binds through F, of HCl through Cl, of methanol through O)."""
    for template, (weaker, stronger) in TEMPLATE_BASES.items():
        molecule = TEMPLATES[template]()
        donor, acceptor = molecule["atoms"][molecule["donor"]][0], molecule["atoms"][molecule["acceptor"]][0]
        weak_atom, strong_atom = (NIST.species[NIST.get(i).species]["basic_atom"] for i in (weaker, stronger))
        assert (weak_atom, strong_atom) == (donor, acceptor), template


def test_a_malformed_set_is_refused(tmp_path, monkeypatch):
    import substrate.datasets as datasets
    (tmp_path / "bad.yaml").write_text("source: {database: d, retrieved: r}\nspecies: {A: {}}\nvalues:\n  - {id: x, quantity: q, species: B, value: 1}\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "_DATA", tmp_path)
    with pytest.raises(ValidationError, match="unknown species 'B'"):
        datasets.load_reference_set("bad")
    (tmp_path / "dup.yaml").write_text("source: {database: d, retrieved: r}\nspecies: {A: {}}\nvalues:\n  - {id: x, quantity: q, species: A, value: 1}\n"
                                       "  - {id: x, quantity: q, species: A, value: 2}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate value id 'x'"):
        datasets.load_reference_set("dup")
    with pytest.raises(ValidationError, match="no reference set 'missing'"):
        datasets.load_reference_set("missing")
