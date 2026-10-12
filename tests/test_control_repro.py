"""Reproducibility records: what counts as the same software, how a git commit is read without a process, and when two sets of results agree."""
import numpy as np
import pytest

from substrate import Quantity
from substrate.control import repro
from substrate.control.repro import (compare_summaries, environment_differences, environment_fingerprint, environment_hash, git_commit, hash_tree,
                                     observables_summary, result_digest)


# -- the environment ----------------------------------------------------------------------------------------------------------------
def test_the_fingerprint_records_the_software_identity_and_the_host():
    fp = environment_fingerprint()
    assert {"substrate", "python_version", "packages", "source_sha256", "git_commit", "image_digest", "platform", "machine", "cpu_count"} <= set(fp)
    assert fp["packages"]["numpy"] and fp["packages"]["scipy"] and len(fp["source_sha256"]) == 64


def test_the_environment_hash_follows_the_software_and_ignores_the_host():
    fp = environment_fingerprint()
    base = environment_hash(fp)
    for host_only in ({"platform": "Plan9"}, {"machine": "riscv"}, {"cpu_count": 3}, {"git_commit": "f" * 40}, {"python_implementation": "PyPy"}):
        assert environment_hash({**fp, **host_only}) == base, host_only
    assert environment_hash({**fp, "python_version": "3.99.0"}) != base
    assert environment_hash({**fp, "substrate": "9.9.9"}) != base
    assert environment_hash({**fp, "source_sha256": "0" * 64}) != base
    assert environment_hash({**fp, "image_digest": "sha256:abc"}) != base
    assert environment_hash({**fp, "packages": {**fp["packages"], "numpy": "0.0.1"}}) != base


def test_the_image_digest_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_IMAGE_DIGEST", "sha256:deadbeef")
    assert environment_fingerprint()["image_digest"] == "sha256:deadbeef"
    monkeypatch.delenv("SUBSTRATE_IMAGE_DIGEST")
    assert environment_fingerprint()["image_digest"] is None


def test_differences_list_every_key_that_changed():
    a = {"x": 1, "y": 2, "z": 3}
    assert environment_differences(a, {"x": 1, "y": 5, "w": 0}) == {"y": (2, 5), "z": (3, None), "w": (None, 0)}
    assert environment_differences(a, dict(a)) == {}


def test_a_source_hash_changes_with_any_file_and_ignores_caches(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    base = hash_tree(tmp_path)
    assert hash_tree(tmp_path) == base
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.py").write_text("junk")
    (tmp_path / "notes.txt").write_text("not python")
    assert hash_tree(tmp_path) == base                                                     # caches and non-Python files do not count
    (tmp_path / "sub" / "b.py").write_text("y = 3\n")
    changed = hash_tree(tmp_path)
    assert changed != base
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    (tmp_path / "sub" / "b.py").rename(tmp_path / "sub" / "c.py")
    assert hash_tree(tmp_path) not in (base, changed)                                     # a rename is a change
    (tmp_path / "sub" / "c.py").rename(tmp_path / "sub" / "b.py")
    (tmp_path / "new.py").write_text("")
    assert hash_tree(tmp_path) != base                                                     # so is a new empty file


def test_the_package_source_hash_is_stable_within_a_process():
    assert repro.source_hash() == repro.source_hash() and len(repro.source_hash()) == 64


# -- the git commit, read without starting a process --------------------------------------------------------------------------------
SHA = "0123456789abcdef0123456789abcdef01234567"


def make_git(root, head, refs=None, packed=None):
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text(head)
    for ref, value in (refs or {}).items():
        (git / ref).parent.mkdir(parents=True, exist_ok=True)
        (git / ref).write_text(value + "\n")
    if packed:
        (git / "packed-refs").write_text("# pack-refs\n" + packed)
    return git


def test_a_branch_checkout_is_read_through_its_loose_ref(tmp_path):
    make_git(tmp_path, "ref: refs/heads/main\n", {"refs/heads/main": SHA})
    (tmp_path / "pkg").mkdir()
    assert git_commit(tmp_path / "pkg" / "mod.py") == SHA                                  # found from a file deeper in the tree


def test_a_packed_ref_is_found_when_there_is_no_loose_one(tmp_path):
    make_git(tmp_path, "ref: refs/heads/main\n", packed=f"{'f' * 40} refs/heads/other\n{SHA} refs/heads/main\n")
    assert git_commit(tmp_path / "x.py") == SHA


def test_a_detached_head_is_the_commit_itself_and_junk_is_none(tmp_path):
    git = make_git(tmp_path, SHA + "\n")
    assert git_commit(tmp_path / "x.py") == SHA
    (git / "HEAD").write_text("not-a-commit\n")
    assert git_commit(tmp_path / "x.py") is None


def test_a_missing_ref_and_a_directory_that_is_not_a_checkout_give_none(tmp_path):
    make_git(tmp_path, "ref: refs/heads/gone\n")
    assert git_commit(tmp_path / "x.py") is None
    elsewhere = tmp_path / "plain"
    elsewhere.mkdir()
    assert git_commit(elsewhere / "x.py") is None


def test_this_checkout_has_a_commit():
    commit = git_commit()
    assert commit is None or (len(commit) == 40 and int(commit, 16) >= 0)


# -- results --------------------------------------------------------------------------------------------------------------------------
def summary(**named):
    return observables_summary({k: Quantity(v, "eV") for k, v in named.items()})


def test_observables_are_flattened_with_unit_and_shape():
    s = observables_summary({"scalar": Quantity(2.5, "eV"), "grid": Quantity(np.arange(6.0).reshape(2, 3), "angstrom")})
    assert s["scalar"] == {"unit": "eV", "shape": [], "values": [2.5]}
    assert s["grid"] == {"unit": "angstrom", "shape": [2, 3], "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}


def test_identical_results_match_with_zero_difference():
    a = summary(barrier=0.5, surface=np.linspace(0, 1, 50))
    c = compare_summaries(a, summary(barrier=0.5, surface=np.linspace(0, 1, 50)))
    assert c.matches and c.max_relative_difference == 0.0 and c.differing == [] and set(c.worst) == {"barrier", "surface"}


def test_noise_inside_the_tolerance_matches_and_outside_it_does_not():
    a = summary(x=1.0)
    assert compare_summaries(a, summary(x=1.0 + 5e-9)).matches
    outside = compare_summaries(a, summary(x=1.0 + 5e-7))
    assert not outside.matches and outside.differing == ["x"] and outside.max_relative_difference == pytest.approx(5e-7, rel=1e-3)
    assert compare_summaries(a, summary(x=1.0 + 5e-7), rtol=1e-5).matches                  # the tolerance is the caller's to set


def test_the_comparison_is_relative_so_large_and_small_values_are_treated_alike():
    assert compare_summaries(summary(x=1e12), summary(x=1e12 * (1 + 1e-9))).matches
    assert not compare_summaries(summary(x=1e-12), summary(x=2e-12)).matches


def test_a_difference_is_not_swallowed_just_because_the_value_is_small():
    """A relaxation time of ~9e-11 s that is off by 0.1% is a real disagreement; an absolute tolerance of 1e-12 would have called it a match."""
    assert not compare_summaries(summary(t=8.80e-11), summary(t=8.81e-11)).matches
    assert compare_summaries(summary(t=8.80e-11), summary(t=8.80e-11 * (1 + 1e-10))).matches


def test_noise_in_the_tail_of_an_array_is_tolerated_but_a_real_change_in_it_is_not():
    base = np.array([1.0, 1e-3, 1e-17])
    assert compare_summaries(summary(x=base), summary(x=np.array([1.0, 1e-3, 3e-17]))).matches                # 1e-17 against 3e-17 is rounding noise next to 1.0
    assert not compare_summaries(summary(x=base), summary(x=np.array([1.0, 2e-3, 1e-17]))).matches            # but 1e-3 against 2e-3 is not
    assert not compare_summaries(summary(x=np.array([1e-3, 2e-3])), summary(x=np.array([1e-3, 3e-3]))).matches


def test_one_wrong_element_in_an_array_is_found_and_the_worst_is_reported():
    base = np.linspace(1, 2, 100)
    wrong = base.copy()
    wrong[63] *= 1.01
    c = compare_summaries(summary(a=base, b=base), summary(a=wrong, b=base))
    assert c.differing == ["a"] and c.worst["a"] == pytest.approx(0.01 / 1.01, rel=1e-6) and c.worst["b"] == 0.0


def test_missing_names_shape_and_unit_changes_all_count_as_a_mismatch():
    a = summary(x=1.0, y=2.0)
    c = compare_summaries(a, summary(x=1.0, z=3.0))
    assert not c.matches and c.only_in_original == ["y"] and c.only_in_reproduction == ["z"]
    reshaped = compare_summaries(summary(x=np.zeros(4)), summary(x=np.zeros((2, 2))))
    assert not reshaped.matches and reshaped.differing == ["x"]
    other_unit = compare_summaries({"x": {"unit": "eV", "shape": [], "values": [1.0]}}, {"x": {"unit": "Ha", "shape": [], "values": [1.0]}})
    assert not other_unit.matches


def test_nans_must_be_in_the_same_places_and_equal_infinities_agree():
    assert compare_summaries(summary(x=np.array([1.0, np.nan])), summary(x=np.array([1.0, np.nan]))).matches
    assert not compare_summaries(summary(x=np.array([1.0, np.nan])), summary(x=np.array([np.nan, 1.0]))).matches
    assert compare_summaries(summary(x=np.array([np.inf, 1.0])), summary(x=np.array([np.inf, 1.0]))).matches
    assert not compare_summaries(summary(x=np.array([np.inf])), summary(x=np.array([1e300]))).matches
    assert not compare_summaries(summary(x=np.array([np.inf])), summary(x=np.array([-np.inf]))).matches


def test_the_digest_ignores_last_bit_noise_and_notices_real_change():
    a = summary(x=np.linspace(0, 1, 20) + 1.0)
    assert result_digest(a) == result_digest(summary(x=np.linspace(0, 1, 20) + 1.0 + 1e-15))
    assert result_digest(a) != result_digest(summary(x=np.linspace(0, 1, 20) + 1.0 + 1e-6))
    assert result_digest(summary(x=1.0)) != result_digest(summary(y=1.0))                  # the name is part of it
    assert result_digest(summary(x=np.zeros(4))) != result_digest(summary(x=np.zeros((2, 2))))


def test_the_comparison_can_be_stored_as_json():
    import json
    c = compare_summaries(summary(x=1.0), summary(x=1.1))
    assert json.loads(json.dumps(c.to_dict()))["differing"] == ["x"]
