"""Cases of the reproducibility records that the first tests did not pin, found by mutation testing: the digest distinguishes what it should, the comparison reports what it found, and
the source hash and the git lookup do not depend on accidents of listing order or of a name that merely contains another."""
import numpy as np
import pytest

from substrate import Quantity
from substrate.control.repro import compare_summaries, git_commit, hash_tree, observables_summary, result_digest

from test_control_repro import SHA, make_git, summary


def raw(unit="eV", **named):
    """A summary built by hand, so that NaN and infinity can be put exactly where a test wants them."""
    return {name: {"unit": unit, "shape": [len(values)], "values": list(values)} for name, values in named.items()}


nan, inf = float("nan"), float("inf")


# -- the digest -----------------------------------------------------------------------------------------------------------------------
def test_the_digest_tells_nan_positive_infinity_and_negative_infinity_apart():
    digests = {result_digest(raw(a=[v])) for v in (nan, inf, -inf, 1.0)}
    assert len(digests) == 4


def test_the_digest_includes_the_unit():
    assert result_digest(summary(x=1.0)) != result_digest(observables_summary({"x": Quantity(1.0, "kJ/mol")}))


# -- the comparison -------------------------------------------------------------------------------------------------------------------
def test_a_number_where_the_original_has_nan_is_a_difference_even_when_everything_else_agrees():
    result = compare_summaries(raw(a=[1.0, nan]), raw(a=[1.0, 5.0]))
    assert not result.matches and result.differing == ["a"] and result.max_relative_difference == inf
    assert not compare_summaries(raw(a=[1.0, 5.0]), raw(a=[1.0, nan])).matches


def test_an_observable_that_is_nothing_but_nan_matches_itself():
    result = compare_summaries(raw(a=[nan, nan], b=[1.0]), raw(a=[nan, nan], b=[1.0]))
    assert result.matches and result.differing == [] and result.max_relative_difference == 0.0


def test_a_finite_value_against_an_infinite_one_is_infinitely_far_apart_and_is_reported_so():
    result = compare_summaries(raw(a=[1.0]), raw(a=[inf]))
    assert not result.matches and result.max_relative_difference == inf and result.worst == {"a": inf}
    assert compare_summaries(raw(a=[inf]), raw(a=[inf])).matches


def test_small_elements_are_held_to_the_tolerance_unless_they_are_below_the_floor():
    big_and_small = raw(a=[1.0, 1e-3])
    assert not compare_summaries(big_and_small, raw(a=[1.0, 1.5e-3]), rtol=1e-3).matches        # a 50% change in an element a thousandth of the largest is a difference
    assert compare_summaries(raw(a=[1.0, 1e-9]), raw(a=[1.0, 1.5e-9]), rtol=1e-3).matches        # but a change of 5e-10 in an element a billionth of the largest is under rtol x floor x largest: noise in a tail
    assert not compare_summaries(raw(a=[1.0, 1e-9]), raw(a=[1.0, 5e-9]), rtol=1e-3).matches      # and a change of 4e-9 there is not


def test_observables_present_in_only_one_run_are_a_mismatch_whichever_run_has_them():
    only_original = compare_summaries(raw(a=[1.0], b=[2.0]), raw(a=[1.0]))
    only_reproduction = compare_summaries(raw(a=[1.0]), raw(a=[1.0], b=[2.0]))
    assert not only_original.matches and only_original.only_in_original == ["b"] and only_original.only_in_reproduction == []
    assert not only_reproduction.matches and only_reproduction.only_in_reproduction == ["b"] and only_reproduction.only_in_original == []


def test_the_overall_difference_is_the_largest_of_all_the_observables_not_the_last():
    result = compare_summaries(raw(a=[1.0], b=[1.0], c=[1.0]), raw(a=[1.1], b=[1.0], c=[1.0000001]))
    assert result.max_relative_difference == pytest.approx(0.1 / 1.1, rel=1e-9) and result.worst["a"] == result.max_relative_difference
    assert result.worst["b"] == 0.0 and 0 < result.worst["c"] < 1e-6


# -- the source hash and the git lookup ------------------------------------------------------------------------------------------------
def test_the_source_hash_does_not_depend_on_the_order_the_file_system_lists_files(tmp_path, monkeypatch):
    from pathlib import Path
    for name in ("a.py", "b.py", "sub/c.py"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(f"x = '{name}'\n")
    expected = hash_tree(tmp_path)
    real = Path.rglob
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(reversed(list(real(self, pattern)))))       # a file system that lists them the other way round
    assert hash_tree(tmp_path) == expected


def test_a_packed_ref_is_matched_by_its_whole_name(tmp_path):
    make_git(tmp_path, "ref: refs/heads/main\n", packed=f"{'e' * 40} refs/heads/main-old\n{'d' * 40} refs/remotes/origin/refs/heads/main\n{SHA} refs/heads/main\n")
    assert git_commit(tmp_path / "x.py") == SHA
