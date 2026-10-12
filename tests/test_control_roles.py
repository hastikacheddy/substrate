"""The whole role table and the shape of an API key, written out. A change to who may do what is a security decision, so it has to be made here as well as in `security.py`; the spot checks in
`test_control_registry.py` could not tell when an entry drifted (mutation testing changed two and nothing failed)."""
import hashlib

import pytest

from substrate.control.security import (Action, DEFAULT_QUOTAS, KEY_PREFIX, PERMISSIONS, PER_TENANT_NOT_PER_DAY, Role, hash_secret, new_api_key, secrets_match,
                                        split_token)

A = Action
EXPECTED = {
    Role.ADMIN: set(Action),
    Role.RESEARCHER: {A.EXPERIMENT_READ, A.EXPERIMENT_CREATE, A.RUN_READ, A.RUN_CREATE, A.RUN_CANCEL, A.RUN_REPRODUCE, A.ARTIFACT_READ, A.DATASET_READ, A.DATASET_INGEST,
                      A.TRANSFER_RUN, A.QPU_USE},
    Role.OPERATOR: {A.RUN_READ, A.RUN_CANCEL, A.DATASET_READ, A.METRICS_READ, A.AUDIT_READ},
    Role.VIEWER: {A.EXPERIMENT_READ, A.RUN_READ, A.ARTIFACT_READ, A.DATASET_READ},
    Role.SERVICE: {A.QUEUE_WORK},
}


@pytest.mark.parametrize("role", list(Role))
def test_each_role_may_do_exactly_these_things(role):
    assert set(PERMISSIONS[role]) == EXPECTED[role], sorted(a.value for a in set(PERMISSIONS[role]) ^ EXPECTED[role])


def test_every_role_and_every_action_is_covered_by_the_table():
    assert set(PERMISSIONS) == set(Role) == set(EXPECTED)
    granted = set().union(*PERMISSIONS.values())
    assert granted == set(Action)                                                       # no action exists that nobody can do


def test_only_administrators_administer_and_only_operators_and_administrators_see_metrics_and_the_audit_trail():
    for action in (A.ADMIN, A.METRICS_READ, A.AUDIT_READ):
        allowed = {role for role in Role if action in PERMISSIONS[role]}
        assert allowed == ({Role.ADMIN} if action is A.ADMIN else {Role.ADMIN, Role.OPERATOR}), action
    assert {role for role in Role if A.QPU_USE in PERMISSIONS[role]} == {Role.ADMIN, Role.RESEARCHER}
    assert {role for role in Role if A.ARTIFACT_READ in PERMISSIONS[role]} == {Role.ADMIN, Role.RESEARCHER, Role.VIEWER}          # an operator never reads the data itself


def test_the_default_allowances_are_the_documented_ones():
    assert DEFAULT_QUOTAS == {"runs_per_day": 200, "tasks_per_day": 5_000, "compute_seconds_per_day": 32 * 3600, "qpu_jobs_per_day": 0, "concurrent_runs": 4,
                              "artifact_bytes_total": 50 * 1024**3, "dataset_ingests_per_day": 50, "experiments_total": 10_000}
    assert PER_TENANT_NOT_PER_DAY == {"concurrent_runs", "artifact_bytes_total", "experiments_total"}


# -- keys ---------------------------------------------------------------------------------------------------------------------------
def test_a_key_carries_256_random_bits_and_only_the_sha256_of_its_secret_is_kept():
    token, stored = new_api_key("K00007")
    prefix, key_id, secret = token.split("_", 2)
    assert (prefix, key_id) == ("sbs", "K00007") == (KEY_PREFIX, "K00007")
    assert len(secret) >= 43                                                      # 32 random bytes in URL-safe base64
    assert stored == hashlib.sha256(secret.encode()).hexdigest() == hash_secret(secret) and secret not in stored and len(stored) == 64
    assert secrets_match(secret, stored) and not secrets_match(secret + "x", stored) and not secrets_match(secret, stored[:-1] + ("0" if stored[-1] != "0" else "1"))
    assert new_api_key("K00007")[0] != token                                      # every key is different


@pytest.mark.parametrize("token, expected", [("sbs_K00001_abc", ("K00001", "abc")), ("sbs_K00001_ab_cd-ef_gh", ("K00001", "ab_cd-ef_gh")),
                                             ("key_K00001_abc", None), ("sbs_K00001_", None), ("sbs__abc", None), ("sbs_K00001", None), ("", None), ("sbs", None)])
def test_a_token_is_understood_only_in_its_exact_shape_and_a_secret_may_contain_underscores(token, expected):
    assert split_token(token) == expected
