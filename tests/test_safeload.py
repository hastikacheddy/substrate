"""Experiment files are not trusted: what the loader refuses, and that it still reads everything this repository ships."""
import pytest

from substrate import ResourceLimitError, ValidationError, load_experiment
from substrate import safeload
from substrate.safeload import load_json, load_yaml, read_spec_file, read_text_file

from conftest import EXPERIMENTS

BILLION_LAUGHS = """
a: &a [x, x, x, x, x, x, x, x, x]
b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a]
c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b]
d: &d [*c, *c, *c, *c, *c, *c, *c, *c, *c]
e: [*d, *d, *d, *d, *d, *d, *d, *d, *d]
"""


def test_a_document_that_expands_by_aliases_is_refused_without_being_expanded():
    with pytest.raises(ValidationError, match="aliases"):
        load_yaml(BILLION_LAUGHS, "bomb.yaml")


def test_a_lone_anchor_and_ordinary_documents_are_fine():
    assert load_yaml("a: &x 1\nb: 2\n") == {"a": 1, "b": 2}
    assert load_yaml("{a: [1, 2, {b: c}]}") == {"a": [1, 2, {"b": "c"}]}


def test_deep_nesting_is_refused_at_the_limit_and_not_before():
    ok = "[" * safeload.MAX_DEPTH + "]" * safeload.MAX_DEPTH
    assert load_yaml(ok) is not None
    with pytest.raises(ValidationError, match="levels deep"):
        load_yaml("[" * (safeload.MAX_DEPTH + 1) + "]" * (safeload.MAX_DEPTH + 1), "deep.yaml")


def test_many_sibling_containers_are_not_mistaken_for_deep_nesting():
    """Depth is the nesting level, not the number of containers seen: a hundred lists side by side are two levels deep."""
    document = "[" + ", ".join("[1, {a: 2}]" for _ in range(100)) + "]"
    assert len(load_yaml(document)) == 100


def test_a_document_with_too_many_nodes_is_refused(monkeypatch):
    monkeypatch.setattr(safeload, "MAX_EVENTS", 50)
    assert load_yaml("[" + ",".join(["1"] * 20) + "]")
    with pytest.raises(ValidationError, match="YAML events"):
        load_yaml("[" + ",".join(["1"] * 60) + "]", "long.yaml")


def test_malformed_yaml_and_json_are_validation_errors_that_name_the_file():
    with pytest.raises(ValidationError, match=r"bad\.yaml: not valid YAML"):
        load_yaml("a: [1, 2\nb: {", "bad.yaml")
    with pytest.raises(ValidationError, match=r"bad\.json: not valid JSON"):
        load_json('{"a": ', "bad.json")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_constants_that_are_not_numbers_an_experiment_can_use_are_refused(constant):
    with pytest.raises(ValidationError, match="not valid JSON"):
        load_json('{"value": %s}' % constant, "x.json")
    assert load_json('{"value": 1.5}') == {"value": 1.5}


def test_deeply_nested_json_is_refused_not_a_crash():
    with pytest.raises(ValidationError, match="nested too deeply"):
        load_json("[" * 200_000, "deep.json")


def test_a_file_above_the_size_limit_is_refused_before_it_is_read(tmp_path, monkeypatch):
    path = tmp_path / "big.yaml"
    path.write_text("a: " + "x" * 5000, encoding="utf-8")
    assert read_text_file(path)
    monkeypatch.setenv("SUBSTRATE_MAX_SPEC_BYTES", "1000")
    with pytest.raises(ResourceLimitError, match=r"big\.yaml: file size in bytes"):
        read_text_file(path)
    with pytest.raises(ResourceLimitError):
        load_experiment(path)


def test_the_default_file_size_limit_is_one_megabyte(tmp_path):
    path = tmp_path / "edge.yaml"
    path.write_bytes(b"#" * 1_000_000)
    assert read_text_file(path)
    path.write_bytes(b"#" * 1_000_001)
    with pytest.raises(ResourceLimitError, match="1,000,001"):
        read_text_file(path)


def test_files_are_read_as_yaml_or_json_by_their_extension(tmp_path):
    (tmp_path / "a.yaml").write_text("k: 1\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("k: 2\n", encoding="utf-8")
    (tmp_path / "c.json").write_text('{"k": 3}', encoding="utf-8")
    assert [read_spec_file(tmp_path / n)["k"] for n in ("a.yaml", "b.yml", "c.json")] == [1, 2, 3]


def test_an_experiment_file_with_an_alias_is_refused_by_the_loader(tmp_path):
    path = tmp_path / "alias.yaml"
    path.write_text("system: &s {name: w}\nagain: *s\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="aliases"):
        load_experiment(path)


def test_every_experiment_this_repository_ships_still_loads():
    paths = sorted(EXPERIMENTS.rglob("*.yaml"))
    assert len(paths) >= 30
    for path in paths:
        assert load_experiment(path).system.parameters is not None, path.name
