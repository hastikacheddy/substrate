"""Coarse-graining throws information away. Each translator says what it drops, the pipeline records it hop by hop, and old records without the field still load."""
import json

import pytest

from substrate import Pipeline, default_registry
from substrate.ir import ProvenanceRecord, ScientificSystem
from substrate.experiment import parse_experiment, read_spec

from conftest import EXPERIMENTS


def test_every_translator_states_what_it_discards_specifically():
    translators = default_registry().translators
    assert len(translators) >= 6
    seen = set()
    for translator in translators:
        assert translator.discarded, f"{translator.name} does not say what it discards"
        for item in translator.discarded:
            assert len(item) > 40 and item == item.strip() and not item.endswith("."), (translator.name, item)       # a statement, not a label
            assert item not in seen, f"'{item}' is repeated across translators"
            seen.add(item)
        assert set(translator.discarded).isdisjoint(translator.approximations)                                       # what is dropped is not an assumption in disguise


def test_the_declared_loss_is_not_generic_each_names_something_of_its_own_hop():
    by_name = {t.name: t for t in default_registry().translators}
    assert any("excited-state" in d for d in by_name["electronic_to_quantum.born_oppenheimer"].discarded)
    assert any("donor-acceptor distance" in d for d in by_name["electronic_to_quantum.born_oppenheimer"].discarded)
    assert any("tunnelling parameter" in d for d in by_name["molecular_to_reaction.tst"].discarded)
    assert any("coherent" in d for d in by_name["quantum_to_reaction.sequential_tunneling"].discarded)
    assert any("intermediate states" in d for d in by_name["biophysical_to_pathway.enzyme_in_pathway"].discarded)
    assert any("time course" in d for d in by_name["reaction_to_enzyme.chemical_step"].discarded)


def test_a_run_records_the_loss_at_each_translation_and_none_at_a_solve():
    experiment = parse_experiment(read_spec(EXPERIMENTS / "proton_transfer_pathway.yaml"))
    final = Pipeline(default_registry()).run(experiment.system, experiment.propagation).final
    translations = [r for r in final.provenance if r.step == "translate"]
    solves = [r for r in final.provenance if r.step == "solve"]
    assert len(translations) == 4 and len(solves) == 5
    registry = {t.name: t for t in default_registry().translators}
    for record in translations:
        assert record.discarded == list(registry[record.name].discarded) and record.discarded
    assert all(record.discarded == [] for record in solves)


def test_the_loss_survives_serialisation_and_a_record_without_the_field_still_loads():
    record = ProvenanceRecord("translate", "t", "1", "a", "b", "x", "y", discarded=["something"])
    assert ProvenanceRecord(**json.loads(json.dumps(record.to_dict()))).discarded == ["something"]
    old = {k: v for k, v in record.to_dict().items() if k != "discarded"}                    # a stored system from before the field existed
    assert ProvenanceRecord(**old).discarded == []
    system = ScientificSystem.from_dict({"name": "n", "scale": "quantum", "kind": "k", "provenance": [old]})
    assert system.provenance[0].discarded == []


def test_the_fingerprint_does_not_depend_on_what_was_discarded():
    experiment = parse_experiment(read_spec(EXPERIMENTS / "proton_transfer.yaml"))
    final = Pipeline(default_registry()).run(experiment.system, experiment.propagation).final
    before = final.fingerprint()
    final.provenance[0].discarded = ["edited"]
    assert final.fingerprint() == before                                                      # lineage identifies content, not commentary
