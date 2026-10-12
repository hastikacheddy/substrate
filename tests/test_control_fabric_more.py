"""Cases of the data adapters that the first tests did not pin, found by mutation testing: the corners of record validation (a pattern must match the whole value, a limit includes
its own value, a record must be an object) and of reading each source's response (an odd shape is skipped or refused, never half-read)."""
import json

import pytest

from substrate.errors import ValidationError
from substrate.control.fabric.adapters import (Field, MAX_RECORDS, NistIonEnergetics, PubChemCompound, RcsbEntry, _slug, validate_records)


# -- validate_records --------------------------------------------------------------------------------------------------------------------
def test_exactly_the_maximum_number_of_records_is_accepted_and_one_more_is_not():
    validate_records([{}] * MAX_RECORDS, ())
    with pytest.raises(ValidationError, match="100,001 records"):
        validate_records([{}] * (MAX_RECORDS + 1), ())
    assert MAX_RECORDS == 100_000


def test_a_record_must_be_an_object_and_not_a_list_or_a_string():
    for bad in ([1, 2], "text", 5, None):
        with pytest.raises(ValidationError, match="record 0 is not an object"):
            validate_records([bad], (Field("a", "int", required=False),))


def test_an_error_names_at_most_five_problems_however_many_a_record_has():
    fields = tuple(Field(f"f{i}", "str") for i in range(9))
    with pytest.raises(ValidationError) as caught:
        validate_records([{}], fields, owner="d")
    message = str(caught.value)
    assert message.count("is missing") == 5 and "f4" in message and "f5" not in message


def test_a_pattern_has_to_match_the_whole_value_and_a_limit_includes_its_own_value():
    pdb = (Field("id", "str", pattern=r"[0-9][A-Z0-9]{3}"),)
    validate_records([{"id": "4HHB"}], pdb)
    for bad in ("4HHBX", "X4HH", "4hhb", ""):
        with pytest.raises(ValidationError, match="expected form"):
            validate_records([{"id": bad}], pdb)
    ranged = (Field("x", "float", minimum=0.0, maximum=10.0),)
    validate_records([{"x": 0.0}, {"x": 10.0}, {"x": 5}], ranged)
    for bad, word in ((-0.001, "below"), (10.001, "above")):
        with pytest.raises(ValidationError, match=word):
            validate_records([{"x": bad}], ranged)


def test_a_dataset_name_made_into_an_id_is_cut_at_forty_characters():
    assert _slug("A Very, Very Long Name " * 5) == _slug("A Very, Very Long Name " * 5)[:40] and len(_slug("x" * 100)) == 40
    assert PubChemCompound().dataset_id({"name": "a" * 100}) == "pubchem.compound." + "a" * 40
    assert _slug("!!!") == "x"


# -- reading responses ------------------------------------------------------------------------------------------------------------------
def test_not_a_number_in_a_response_is_refused_not_stored():
    body = b'{"PropertyTable": {"Properties": [{"CID": 1, "MolecularFormula": "H2O", "MolecularWeight": NaN, "InChIKey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N"}]}}'
    with pytest.raises(ValidationError, match="not valid JSON"):
        PubChemCompound().parse(body)
    with pytest.raises(ValidationError, match="not valid JSON"):
        RcsbEntry().parse(b'{"rcsb_id": "4HHB", "struct": {"title": Infinity}}')


RCSB = {"rcsb_id": "4HHB", "struct": {"title": "HEMOGLOBIN"}}


@pytest.mark.parametrize("extra, absent", [
    ({"exptl": [5]}, "method"), ({"exptl": ["method"]}, "method"), ({"exptl": [{"no_method": 1}]}, "method"), ({"exptl": []}, "method"),
    ({"rcsb_entry_info": {"resolution_combined": 2.0}}, "resolution_angstrom"), ({"rcsb_entry_info": {"resolution_combined": []}}, "resolution_angstrom"),
    ({"rcsb_accession_info": {"initial_release_date": "2020"}}, "release_date"), ({"rcsb_accession_info": {"initial_release_date": 20200101}}, "release_date"),
])
def test_an_odd_optional_part_of_an_rcsb_entry_is_left_out_rather_than_misread_or_fatal(extra, absent):
    (record,) = RcsbEntry().parse(json.dumps({**RCSB, **extra}).encode())
    assert absent not in record and record["id"] == "4HHB" and record["title"] == "HEMOGLOBIN"


def test_the_good_optional_parts_of_an_rcsb_entry_are_read():
    full = {**RCSB, "exptl": [{"method": "X-RAY DIFFRACTION"}], "rcsb_entry_info": {"resolution_combined": [1.74], "polymer_entity_count": 2},
            "rcsb_accession_info": {"initial_release_date": "1984-07-17T00:00:00+0000"}}
    (record,) = RcsbEntry().parse(json.dumps(full).encode())
    assert record == {"id": "4HHB", "title": "HEMOGLOBIN", "method": "X-RAY DIFFRACTION", "resolution_angstrom": 1.74, "release_date": "1984-07-17", "polymer_entity_count": 2}


def test_empty_optional_values_in_the_bundled_set_are_dropped_before_validation():
    adapter = NistIonEnergetics()
    body = json.dumps({"records": [{"id": "x", "quantity": "q", "species": "s", "value": 1.0, "unit": "kJ/mol", "uncertainty": None, "method": "", "reference": "r"}]}).encode()
    assert adapter.parse(body) == [{"id": "x", "quantity": "q", "species": "s", "value": 1.0, "unit": "kJ/mol", "reference": "r"}]
    shipped = adapter.parse(adapter.read())
    assert shipped and all(v not in (None, "") for row in shipped for v in row.values())
    validate_records(shipped, adapter.fields, owner="nist")
