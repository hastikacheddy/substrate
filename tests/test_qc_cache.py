"""The energy cache: what is stored is sealed with a digest, a damaged or foreign entry is not served, and the cache can be audited.

A stale or hand-edited energy is the worst kind of wrong number: it looks exactly like a computed one. The digest catches damage and
accidental overwrites (a half-written row, an edit by hand, results left behind by a buggy version of the worker). It does not stop a user
who can write the database from re-sealing an entry, and it is not meant to: the cache is the user's own file.
"""
import json
import sqlite3

import pytest

from substrate.cli import main
from substrate.qc import QCCache, QCJob, QCProgram, QCResult, compute_cached, worker_digest

WATER = (("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692))


class Counting(QCProgram):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def compute(self, jobs):
        self.calls += len(jobs)
        return [QCResult(-76.0 - 0.001 * j.atoms[0][3], True, 0.5, "counting 1") for j in jobs]


def job(z=0.0):
    return QCJob((("H", 0.0, 0.0, z), ("H", 0.0, 0.0, z + 0.8)), 0, 0, "hf", "6-31g")


def raw(cache, key):
    with sqlite3.connect(cache.path) as db:
        return db.execute("SELECT payload FROM energies WHERE key = ?", (key,)).fetchone()[0]


def overwrite(cache, key, payload):
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE energies SET payload = ? WHERE key = ?", (payload, key))


@pytest.fixture
def cache(tmp_path):
    return QCCache(tmp_path / "qc.sqlite3")


def stored_key(program, j):
    return j.key(program.name)


def test_an_entry_is_stored_with_its_digest_and_the_worker_that_made_it_and_read_back_whole(cache):
    result = QCResult(-1.5, True, 0.25, "pyscf 2.14.0", (("H", 0.0, 0.0, 0.1),), -1.4, 0.02, -1.45)
    cache.put_many({"k": result})
    stored = json.loads(raw(cache, "k"))
    assert set(stored) == {"result", "worker", "sha256"} and stored["worker"] == worker_digest()
    assert len(stored["sha256"]) == 64 and stored["result"]["energy"] == -1.5
    assert cache.get_many(["k"]) == {"k": result} and cache.rejected == 0


def test_the_worker_digest_is_a_stable_hash_of_the_worker_source():
    assert worker_digest() == worker_digest() and len(worker_digest()) == 16
    int(worker_digest(), 16)


def test_an_entry_edited_after_it_was_written_is_not_served_and_is_recomputed(cache):
    program, j = Counting(), job()
    compute_cached(program, [j], cache)
    key = stored_key(program, j)
    payload = json.loads(raw(cache, key))
    payload["result"]["energy"] = -999.0                                               # a hand edit; the digest no longer matches
    overwrite(cache, key, json.dumps(payload))
    assert cache.get_many([key]) == {} and cache.rejected == 1
    results, ran, cached = compute_cached(program, [j], cache)
    assert (ran, cached, program.calls) == (1, 0, 2) and results[0].energy_hartree == pytest.approx(-76.0)
    assert cache.get_many([key])[key].energy_hartree == pytest.approx(-76.0)         # the repaired entry is sealed again


def test_garbage_and_non_finite_energies_are_not_served(cache):
    cache.put_many({"a": QCResult(-1.0, True), "b": QCResult(-2.0, True), "c": QCResult(float("nan"), True), "d": QCResult(float("-inf"), True)})
    overwrite(cache, "a", "{not json")
    overwrite(cache, "b", json.dumps({"result": {"energy": -2.0}, "worker": "x", "sha256": "0" * 64}))
    assert cache.get_many(["a", "b", "c", "d"]) == {} and cache.rejected == 4


def test_an_entry_written_before_digests_existed_is_served_and_reported_as_unverified(cache):
    legacy = {"energy": -3.0, "converged": True, "homo_lumo_gap": 0.1, "program": "pyscf 2.14.0"}
    with sqlite3.connect(cache.path) as db:
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("old", json.dumps(legacy)))
    assert cache.get_many(["old"])["old"].energy_hartree == -3.0 and cache.rejected == 0
    assert cache.audit()["unverified"] == 1


def test_strict_mode_serves_only_entries_this_version_of_the_worker_wrote(tmp_path):
    cache = QCCache(tmp_path / "s.sqlite3")
    cache.put_many({"mine": QCResult(-1.0, True)})
    foreign = {"result": QCResult(-2.0, True).to_dict(), "worker": "0123456789abcdef"}
    import hashlib
    foreign["sha256"] = hashlib.sha256(json.dumps(foreign, sort_keys=True).encode()).hexdigest()
    with sqlite3.connect(cache.path) as db:
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("theirs", json.dumps(foreign)))
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("old", json.dumps(QCResult(-3.0, True).to_dict())))
    assert set(cache.get_many(["mine", "theirs", "old"])) == {"mine", "theirs", "old"}              # the default trusts them
    strict = QCCache(tmp_path / "s.sqlite3", strict=True)
    assert set(strict.get_many(["mine", "theirs", "old"])) == {"mine"} and strict.rejected == 2


def test_strict_mode_can_be_switched_on_by_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_QC_CACHE_STRICT", "1")
    assert QCCache(tmp_path / "e.sqlite3").strict
    monkeypatch.setenv("SUBSTRATE_QC_CACHE_STRICT", "0")
    assert not QCCache(tmp_path / "e.sqlite3").strict
    monkeypatch.delenv("SUBSTRATE_QC_CACHE_STRICT")
    assert not QCCache(tmp_path / "e.sqlite3").strict


def test_the_audit_counts_entries_by_state_program_and_worker(cache):
    cache.put_many({"a": QCResult(-1.0, True, None, "pyscf 2.14.0"), "b": QCResult(-2.0, True, None, "pyscf 2.14.0"), "c": QCResult(-3.0, True, None, "pyscf 2.15.0")})
    overwrite(cache, "c", "garbage")
    with sqlite3.connect(cache.path) as db:
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("old", json.dumps(QCResult(-4.0, True, None, "pyscf 2.10.0").to_dict())))
    report = cache.audit()
    assert (report["total"], report["verified"], report["unverified"], report["corrupt"]) == (4, 2, 1, 1)
    assert report["by_program"] == {"pyscf 2.14.0": 2, "pyscf 2.10.0": 1}
    assert report["by_worker"] == {worker_digest(): 2}


def test_purge_removes_corrupt_entries_or_those_of_a_named_worker_or_the_unverified(cache):
    cache.put_many({"a": QCResult(-1.0, True), "b": QCResult(-2.0, True)})
    other = {"result": QCResult(-5.0, True).to_dict(), "worker": "ffffffffffffffff"}
    import hashlib
    other["sha256"] = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()
    with sqlite3.connect(cache.path) as db:
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("w", json.dumps(other)))
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("old", json.dumps(QCResult(-4.0, True).to_dict())))
    overwrite(cache, "b", "garbage")
    assert cache.purge(corrupt=True) == 1 and len(cache) == 3
    assert cache.purge(worker="ffffffffffffffff") == 1 and len(cache) == 2
    assert cache.purge(unverified=True) == 1 and len(cache) == 1
    assert cache.purge() == 0                                                           # nothing asked for, nothing removed
    assert set(cache.get_many(["a"])) == {"a"}


def test_each_purge_selector_removes_only_what_it_names(cache):
    cache.put_many({"good": QCResult(-1.0, True), "bad": QCResult(-2.0, True)})
    overwrite(cache, "bad", "garbage")
    with sqlite3.connect(cache.path) as db:
        db.execute("INSERT INTO energies (key, payload) VALUES (?, ?)", ("old", json.dumps(QCResult(-4.0, True).to_dict())))
    assert cache.purge(unverified=True) == 1 and len(cache) == 2                      # the legacy entry only: the corrupt one stays
    assert cache.audit()["corrupt"] == 1
    assert cache.purge(worker=worker_digest()) == 1 and len(cache) == 1               # the sealed entry only: the corrupt one stays
    assert cache.audit()["corrupt"] == 1
    assert cache.purge(corrupt=True) == 1 and len(cache) == 0


def test_a_refused_request_does_not_even_open_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path / "fresh"))
    bad = QCJob((("H", 0.0, 0.0, 0.0),), 0, 1, "hf", "/etc/passwd")
    with pytest.raises(Exception, match="basis-set name"):
        compute_cached(Counting(), [bad], None)
    assert not (tmp_path / "fresh").exists()


def test_the_command_line_reports_and_cleans_the_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SUBSTRATE_CACHE_DIR", str(tmp_path))
    cache = QCCache()
    cache.put_many({"a": QCResult(-1.0, True, None, "pyscf 2.14.0"), "b": QCResult(-2.0, True, None, "pyscf 2.14.0")})
    overwrite(cache, "b", "garbage")
    assert main(["cache", "status"]) == 0
    out = capsys.readouterr().out
    assert "2 entries" in out and "1 verified" in out and "1 corrupt" in out and "pyscf 2.14.0" in out
    assert f"worker {worker_digest()}: 1  (current)" in out                             # the reader is told which worker is the one running now
    assert main(["cache", "verify"]) == 1                                               # damage found: a non-zero exit, nothing deleted
    assert len(cache) == 2
    assert main(["cache", "verify", "--delete"]) == 0
    assert len(cache) == 1 and main(["cache", "verify"]) == 0
    capsys.readouterr()
    assert main(["cache", "purge"]) == 1 and "choose what to purge" in capsys.readouterr().err
    assert len(cache) == 1                                                              # asking for nothing removes nothing
    cache.put_many({"w": QCResult(-3.0, True)})
    assert main(["cache", "purge", "--worker", worker_digest()]) == 0 and len(cache) == 0
