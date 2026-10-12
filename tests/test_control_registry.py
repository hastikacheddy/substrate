"""The registry: who is who, what each tenant can see, what a role may do, how quotas bite, and that the event log notices tampering."""
import threading

import pytest

sa = pytest.importorskip("sqlalchemy")

from substrate import ResourceLimitError, ValidationError  # noqa: E402
from substrate.control import db  # noqa: E402
from substrate.control.registry import GENESIS, Registry, TaskSpec, sha256_json  # noqa: E402
from substrate.control.security import (AuthorizationError, Action, DEFAULT_QUOTAS, PERMISSIONS, QuotaExceededError, Role,  # noqa: E402
                                        hash_secret, split_token)

SPEC = {"system": {"name": "w", "scale": "quantum", "kind": "quantum.double_well_1d", "parameters": {}}}


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry(tmp_path, clock):
    return Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}", clock=clock)


@pytest.fixture
def lab(registry):
    tenant, admin, token = registry.bootstrap("lab-a")
    researcher = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    viewer = registry.create_principal(tenant, "vera", Role.VIEWER)
    operator = registry.create_principal(tenant, "olga", Role.OPERATOR)
    return {"tenant": tenant, "admin": admin, "token": token, "researcher": researcher, "viewer": viewer, "operator": operator}


def tasks(*names):
    chain = [TaskSpec(names[0], "validate")]
    for before, name in zip(names, names[1:]):
        chain.append(TaskSpec(name, "solve", depends_on=[before]))
    return chain


# -- keys and authentication ---------------------------------------------------------------------------------------------------
def test_a_key_authenticates_its_principal_and_only_the_hash_of_its_secret_is_stored(registry, lab):
    principal = registry.authenticate(lab["token"])
    assert principal == lab["admin"]
    _, secret = split_token(lab["token"])
    with registry.engine.connect() as conn:
        stored = [r[0] for r in conn.execute(sa.select(db.api_keys.c.key_hash))]
        every_text = " ".join(str(v) for row in conn.execute(sa.select(db.api_keys)) for v in row)
    assert stored == [hash_secret(secret)] and len(stored[0]) == 64
    assert secret not in every_text and lab["token"] not in every_text


@pytest.mark.parametrize("bad", ["", "nonsense", "sbs_K00001", "sbs_K00001_", "sbs__secret", "sbs_K99999_" + "x" * 43, None])
def test_malformed_unknown_and_wrong_keys_authenticate_nobody(registry, lab, bad):
    assert registry.authenticate(bad) is None


def test_a_right_id_with_the_wrong_secret_is_refused(registry, lab):
    key_id, _ = split_token(lab["token"])
    assert registry.authenticate(f"sbs_{key_id}_not-the-secret") is None


def test_a_revoked_key_and_a_disabled_principal_stop_working(registry, lab):
    token = registry.issue_api_key(lab["researcher"].id)
    assert registry.authenticate(token).name == "ada"
    key_id, _ = split_token(token)
    assert registry.revoke_api_key(key_id) is True and registry.authenticate(token) is None
    assert registry.revoke_api_key(key_id) is False                                    # already revoked
    other = registry.issue_api_key(lab["viewer"].id)
    with registry.engine.begin() as conn:
        conn.execute(sa.update(db.principals).where(db.principals.c.id == lab["viewer"].id).values(disabled=True))
    assert registry.authenticate(other) is None


def test_every_key_is_different(registry, lab):
    tokens = {registry.issue_api_key(lab["researcher"].id) for _ in range(20)}
    assert len(tokens) == 20


# -- roles ---------------------------------------------------------------------------------------------------------------------
def test_the_role_table_gives_each_role_what_it_needs_and_not_more():
    assert Action.QPU_USE not in PERMISSIONS[Role.VIEWER] and Action.QPU_USE in PERMISSIONS[Role.RESEARCHER]
    assert Action.EXPERIMENT_READ not in PERMISSIONS[Role.OPERATOR] and Action.ARTIFACT_READ not in PERMISSIONS[Role.OPERATOR]
    assert Action.AUDIT_READ in PERMISSIONS[Role.OPERATOR] and Action.RUN_CANCEL in PERMISSIONS[Role.OPERATOR]
    assert PERMISSIONS[Role.SERVICE] == {Action.QUEUE_WORK}
    assert set(PERMISSIONS[Role.ADMIN]) == set(Action)
    assert Action.RUN_CREATE not in PERMISSIONS[Role.VIEWER] and Action.EXPERIMENT_CREATE not in PERMISSIONS[Role.VIEWER]


def test_a_viewer_cannot_create_and_an_operator_cannot_read_experiments(registry, lab):
    with pytest.raises(AuthorizationError, match="vera.*VIEWER.*experiment.create"):
        registry.add_experiment(lab["viewer"], "x", SPEC)
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    assert registry.get_experiment(lab["viewer"], experiment["id"])["name"] == "x"
    with pytest.raises(AuthorizationError):
        registry.get_experiment(lab["operator"], experiment["id"])
    service = registry.create_principal(lab["tenant"], "worker", Role.SERVICE)
    with pytest.raises(AuthorizationError):
        registry.list_runs(service)                                                     # a worker takes tasks; it does not browse runs


# -- tenants cannot see each other -----------------------------------------------------------------------------------------------
def test_one_tenant_cannot_see_or_learn_of_anothers_objects(registry, lab):
    _, boris, _ = registry.bootstrap("lab-b", "boris")
    experiment = registry.add_experiment(lab["researcher"], "private", SPEC)
    run = registry.create_run(lab["researcher"], experiment["id"], seed=1, n_samples=0, environment={}, environment_hash="e", tasks=tasks("a"))
    registry.record_artifact(lab["tenant"], "ab" * 32, run_id=run["id"], name="secret.json", media_type="application/json", size=10)

    assert registry.get_experiment(boris, experiment["id"]) is None                    # the same answer as for an id that does not exist
    assert registry.get_experiment(boris, "EXP-999999") is None
    assert registry.list_experiments(boris) == []
    assert registry.get_run(boris, run["id"]) is None and registry.list_runs(boris) == [] and registry.run_tasks(boris, run["id"]) == []
    assert registry.artifact_meta(boris, "ab" * 32) is None and registry.run_artifacts(boris, run["id"]) == []
    with pytest.raises(LookupError):
        registry.create_run(boris, experiment["id"], seed=1, n_samples=0, environment={}, environment_hash="e", tasks=tasks("a"))
    assert registry.events(f"audit:{lab['tenant']}", boris.tenant_id) == []


def test_the_same_bytes_can_belong_to_many_runs_each_under_its_own_name(registry, lab):
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    first = registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="e", tasks=tasks("a"))["id"]
    second = registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="e", tasks=tasks("a"))["id"]
    registry.record_artifact(lab["tenant"], "ee" * 32, run_id=first, name="check.json", media_type="application/json", size=9)
    registry.record_artifact(lab["tenant"], "ee" * 32, run_id=second, name="same-bytes.json", media_type="application/json", size=9)
    registry.record_artifact(lab["tenant"], "ee" * 32, run_id=second, name="same-bytes.json", media_type="application/json", size=9)        # again: no duplicate
    assert [a["name"] for a in registry.run_artifacts(lab["researcher"], first)] == ["check.json"]
    assert [a["name"] for a in registry.run_artifacts(lab["researcher"], second)] == ["same-bytes.json"]
    assert registry.artifact_bytes_total(lab["tenant"]) == 9                                      # stored once


def test_recording_the_same_artifact_from_many_threads_at_once_does_not_fail(registry, lab):
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    runs = [registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="e", tasks=tasks("a"))["id"]
            for _ in range(4)]
    errors = []

    def record(run_id):
        try:
            for _ in range(5):
                registry.record_artifact(lab["tenant"], "dd" * 32, run_id=run_id, name="shared.json", media_type="application/json", size=7)
        except Exception as error:                                                              # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=record, args=(r,)) for r in runs for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and registry.artifact_bytes_total(lab["tenant"]) == 7
    assert all([a["name"] for a in registry.run_artifacts(lab["researcher"], r)] == ["shared.json"] for r in runs)


def test_the_same_artifact_content_in_two_tenants_is_two_records(registry, lab):
    tenant_b, _, _ = registry.bootstrap("lab-b", "boris")
    registry.record_artifact(lab["tenant"], "cd" * 32, run_id=None, name="a", media_type="application/json", size=5)
    registry.record_artifact(tenant_b, "cd" * 32, run_id=None, name="b", media_type="application/json", size=7)
    assert registry.artifact_bytes_total(lab["tenant"]) == 5 and registry.artifact_bytes_total(tenant_b) == 7


# -- experiments -----------------------------------------------------------------------------------------------------------------
def test_experiments_get_readable_sequential_ids_and_keep_a_hash_of_their_spec(registry, lab):
    first = registry.add_experiment(lab["researcher"], "one", SPEC)
    second = registry.add_experiment(lab["researcher"], "two", {**SPEC, "phenomenon": "other"})
    assert (first["id"], second["id"]) == ("EXP-000001", "EXP-000002")
    assert first["spec_sha256"] == sha256_json(SPEC) != second["spec_sha256"]
    assert first["spec"] == SPEC and first["created_by"] == lab["researcher"].id
    assert [e["id"] for e in registry.list_experiments(lab["viewer"])] == ["EXP-000002", "EXP-000001"]


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"x": float("inf")}, {"x": object()}, {"x": {1, 2}}, {"x": b"bytes"}])
def test_a_spec_that_is_not_plain_json_data_is_refused(registry, lab, bad):
    with pytest.raises((TypeError, ValueError)):
        registry.add_experiment(lab["researcher"], "bad", bad)


def test_the_experiment_quota_is_enforced_per_tenant(registry, lab):
    registry.set_quota(lab["admin"], "experiments_total", 2)
    registry.add_experiment(lab["researcher"], "1", SPEC)
    registry.add_experiment(lab["researcher"], "2", SPEC)
    with pytest.raises(QuotaExceededError, match="already holds 2"):
        registry.add_experiment(lab["researcher"], "3", SPEC)


# -- runs and tasks -----------------------------------------------------------------------------------------------------------------
def test_a_run_gets_its_tasks_with_dependencies_mapped_and_only_the_first_ready(registry, lab):
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    run = registry.create_run(lab["researcher"], experiment["id"], seed=7, n_samples=3, environment={"python": "3"}, environment_hash="h",
                              tasks=[TaskSpec("validate", "validate"), TaskSpec("solve", "solve", depends_on=["validate"], queue="chemistry"),
                                     TaskSpec("report", "check", depends_on=["solve", "validate"])])
    assert run["id"] == "RUN-000001" and run["state"] == "queued" and (run["seed"], run["n_samples"]) == (7, 3)
    rows = {t["name"]: t for t in registry.run_tasks(lab["researcher"], run["id"])}
    assert [t["id"] for t in rows.values()] == ["RUN-000001/00", "RUN-000001/01", "RUN-000001/02"]
    assert rows["validate"]["state"] == "ready" and rows["solve"]["state"] == "pending" and rows["report"]["state"] == "pending"
    assert rows["solve"]["depends_on"] == ["RUN-000001/00"] and rows["report"]["depends_on"] == ["RUN-000001/01", "RUN-000001/00"]
    assert rows["solve"]["queue"] == "chemistry" and rows["validate"]["queue"] == "cpu"


def test_bad_task_graphs_are_refused(registry, lab):
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    with pytest.raises(ValidationError, match="same name"):
        registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="h",
                            tasks=[TaskSpec("a", "validate"), TaskSpec("a", "solve")])
    with pytest.raises(ValidationError, match="unknown task"):
        registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="h",
                            tasks=[TaskSpec("a", "validate", depends_on=["ghost"])])


def test_concurrent_runs_are_limited_per_tenant_and_a_block_is_audited(registry, lab):
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    for _ in range(int(DEFAULT_QUOTAS["concurrent_runs"])):
        registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="h", tasks=tasks("a"))
    with pytest.raises(QuotaExceededError, match="already active"):
        registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="h", tasks=tasks("a"))
    blocked = [e for e in registry.events(f"audit:{lab['tenant']}", lab["tenant"]) if e["kind"] == "quota.blocked"]
    assert len(blocked) == 1 and blocked[0]["payload"]["metric"] == "concurrent_runs" and blocked[0]["payload"]["result"] == "BLOCKED"
    registry.finish_run(lab["tenant"], "RUN-000001", "succeeded", {}, None)            # one finishes: there is room again
    registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="h", tasks=tasks("a"))


def test_a_quota_error_is_a_resource_limit_so_an_ensemble_cannot_drop_it():
    assert issubclass(QuotaExceededError, ResourceLimitError) and not issubclass(QuotaExceededError, ValidationError)


def test_run_state_moves_from_queued_to_running_once(registry, lab, clock):
    experiment = registry.add_experiment(lab["researcher"], "x", SPEC)
    run = registry.create_run(lab["researcher"], experiment["id"], seed=0, n_samples=0, environment={}, environment_hash="h", tasks=tasks("a"))
    clock.t += 5
    registry.mark_run_started(lab["tenant"], run["id"])
    started = registry.get_run(lab["researcher"], run["id"])
    assert started["state"] == "running" and started["started_at"] == clock.t
    clock.t += 5
    registry.mark_run_started(lab["tenant"], run["id"])                                 # a second call changes nothing
    assert registry.get_run(lab["researcher"], run["id"])["started_at"] == started["started_at"]
    registry.finish_run(lab["tenant"], run["id"], "succeeded", {"execution": {"state": "PASS", "reasons": []}}, "digest")
    done = registry.get_run(lab["researcher"], run["id"])
    assert (done["state"], done["result_digest"], done["finished_at"]) == ("succeeded", "digest", clock.t)


# -- usage quotas -------------------------------------------------------------------------------------------------------------------
def test_a_daily_allowance_is_used_up_then_blocked_without_being_used_and_audited(registry, lab):
    registry.set_quota(lab["admin"], "tasks_per_day", 10)
    assert registry.consume(lab["researcher"], "tasks_per_day", 6) == 6
    assert registry.consume(lab["researcher"], "tasks_per_day", 4) == 10
    with pytest.raises(QuotaExceededError, match="tasks_per_day would reach 11 of 10"):
        registry.consume(lab["researcher"], "tasks_per_day", 1)
    assert registry.used_today(lab["researcher"], "tasks_per_day") == 10               # the blocked request used nothing
    event = [e for e in registry.events(f"audit:{lab['tenant']}", lab["tenant"]) if e["kind"] == "quota.blocked"][-1]
    assert event["payload"] == {"principal": lab["researcher"].id, "metric": "tasks_per_day", "requested": 1, "used": 10.0, "limit": 10.0, "result": "BLOCKED"}


def test_allowances_are_per_principal_and_reset_the_next_day(registry, lab, clock):
    registry.set_quota(lab["admin"], "runs_per_day", 1)
    registry.consume(lab["researcher"], "runs_per_day")
    registry.consume(lab["admin"], "runs_per_day")                                      # someone else's allowance is their own
    with pytest.raises(QuotaExceededError):
        registry.consume(lab["researcher"], "runs_per_day")
    clock.t += 24 * 3600
    registry.consume(lab["researcher"], "runs_per_day")


def test_qpu_jobs_are_off_by_default_until_an_administrator_grants_some(registry, lab):
    with pytest.raises(QuotaExceededError):
        registry.consume(lab["researcher"], "qpu_jobs_per_day")
    registry.set_quota(lab["admin"], "qpu_jobs_per_day", 3)
    assert registry.consume(lab["researcher"], "qpu_jobs_per_day") == 1


def test_only_an_administrator_sets_quotas_and_only_known_quotas_with_sane_values(registry, lab):
    with pytest.raises(AuthorizationError):
        registry.set_quota(lab["researcher"], "runs_per_day", 10**6)
    with pytest.raises(KeyError, match="unknown quota"):
        registry.set_quota(lab["admin"], "free_money", 1)
    for bad in (-1, float("nan")):
        with pytest.raises(ValidationError, match="non-negative"):
            registry.set_quota(lab["admin"], "runs_per_day", bad)
    assert registry.quota(lab["tenant"], "runs_per_day") == DEFAULT_QUOTAS["runs_per_day"]
    other, _, _ = registry.bootstrap("lab-c", "carl")
    registry.set_quota(lab["admin"], "runs_per_day", 5)
    assert registry.quota(other, "runs_per_day") == DEFAULT_QUOTAS["runs_per_day"]       # one tenant's override is not another's


# -- the event log --------------------------------------------------------------------------------------------------------------
def test_events_form_a_hash_chain_that_starts_at_a_fixed_point(registry, lab):
    registry.add_experiment(lab["researcher"], "x", SPEC)
    registry.add_experiment(lab["researcher"], "y", SPEC)
    chain = f"audit:{lab['tenant']}"
    events = registry.events(chain, lab["tenant"])
    assert [e["kind"] for e in events] == ["experiment.create", "experiment.create"]
    assert events[0]["prev_hash"] == GENESIS and events[1]["prev_hash"] == events[0]["hash"]
    assert registry.verify_chain(chain, lab["tenant"]) == (True, None)


def _edit_event(registry, seq, **changes):
    with registry.engine.begin() as conn:
        conn.execute(sa.update(db.events).where(db.events.c.seq == seq).values(**changes))


def test_editing_a_past_event_breaks_the_chain_at_that_event(registry, lab):
    for i in range(4):
        registry.audit(lab["tenant"], "note", n=i)
    chain = f"audit:{lab['tenant']}"
    target = registry.events(chain, lab["tenant"])[2]
    _edit_event(registry, target["seq"], payload={"n": 99})
    assert registry.verify_chain(chain, lab["tenant"]) == (False, target["seq"])


def test_deleting_or_reordering_events_breaks_the_chain(registry, lab):
    for i in range(4):
        registry.audit(lab["tenant"], "note", n=i)
    chain = f"audit:{lab['tenant']}"
    events = registry.events(chain, lab["tenant"])
    with registry.engine.begin() as conn:
        conn.execute(sa.delete(db.events).where(db.events.c.seq == events[1]["seq"]))
    intact, where = registry.verify_chain(chain, lab["tenant"])
    assert not intact and where == events[2]["seq"]


def test_a_forged_event_with_a_consistent_hash_but_the_wrong_predecessor_is_caught(registry, lab):
    registry.audit(lab["tenant"], "note", n=1)
    registry.audit(lab["tenant"], "note", n=2)
    chain = f"audit:{lab['tenant']}"
    last = registry.events(chain, lab["tenant"])[-1]
    elsewhere = "e" * 64
    forged = sha256_json([elsewhere, chain, last["ts"], last["kind"], last["payload"]])    # valid for itself, but it points at the wrong predecessor
    _edit_event(registry, last["seq"], prev_hash=elsewhere, hash=forged)
    assert registry.verify_chain(chain, lab["tenant"]) == (False, last["seq"])


def test_the_database_itself_refuses_to_point_a_second_event_at_the_chains_start(registry, lab):
    registry.audit(lab["tenant"], "note", n=1)
    registry.audit(lab["tenant"], "note", n=2)
    last = registry.events(f"audit:{lab['tenant']}", lab["tenant"])[-1]
    with pytest.raises(sa.exc.IntegrityError):
        _edit_event(registry, last["seq"], prev_hash=GENESIS)


def test_chains_are_separate_per_tenant_and_per_run(registry, lab):
    other, _, _ = registry.bootstrap("lab-b", "boris")
    registry.audit(lab["tenant"], "a")
    registry.audit(other, "b")
    assert [e["kind"] for e in registry.events(f"audit:{lab['tenant']}", lab["tenant"])] == ["a"]
    assert registry.events(f"audit:{lab['tenant']}", other) == []
    assert registry.events(f"audit:{other}", other)[0]["prev_hash"] == GENESIS


def test_two_writers_cannot_fork_a_chain(registry, lab):
    registry.audit(lab["tenant"], "base")
    chain = f"audit:{lab['tenant']}"
    last = registry.events(chain, lab["tenant"])[-1]
    with pytest.raises(sa.exc.IntegrityError):                                           # a second child of the same link is refused by the database
        with registry.engine.begin() as conn:
            conn.execute(sa.insert(db.events).values(chain=chain, tenant_id=lab["tenant"], ts=1.0, kind="fork", payload={}, prev_hash=last["prev_hash"], hash="f" * 64))


def test_concurrent_writers_all_get_in_and_the_chain_stays_intact(registry, lab):
    errors = []

    def write(n):
        try:
            for i in range(5):
                registry.audit(lab["tenant"], "note", writer=n, i=i)
        except Exception as error:                                                      # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(registry.events(f"audit:{lab['tenant']}", lab["tenant"])) == 30
    assert registry.verify_chain(f"audit:{lab['tenant']}", lab["tenant"]) == (True, None)


def test_concurrent_creators_never_get_the_same_identifier(registry, lab):
    ids, errors = [], []

    def create(n):
        try:
            for i in range(4):
                ids.append(registry.add_experiment(lab["researcher"], f"{n}-{i}", {**SPEC, "n": n, "i": i})["id"])
        except Exception as error:                                                      # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=create, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(ids) == 24 and len(set(ids)) == 24


# -- starting on PostgreSQL ---------------------------------------------------------------------------------------------------------------
def test_on_postgresql_the_schema_is_created_under_an_advisory_lock_so_pods_starting_together_do_not_race(monkeypatch):
    """No server is needed: a stand-in engine that says it is PostgreSQL records what the registry runs on it."""
    import contextlib
    from substrate.control.registry import SCHEMA_LOCK_KEY
    events = []

    class Conn:
        def execute(self, statement, params=None):
            events.append(("sql", str(statement), params))
            return type("R", (), {"first": lambda self: object(), "scalar_one": lambda self: 0})()

    conn = Conn()

    class Engine:
        dialect = type("D", (), {"name": "postgresql"})()

        @contextlib.contextmanager
        def begin(self):
            events.append(("begin",))
            yield conn
            events.append(("commit",))

    monkeypatch.setattr(db.metadata, "create_all", lambda bind: events.append(("create_all", bind is conn)))
    registry = Registry.__new__(Registry)
    registry.engine = Engine()
    registry._create_schema()
    assert events[0] == ("begin",) and events[1][0] == "sql" and "pg_advisory_xact_lock" in events[1][1] and events[1][2] == {"key": SCHEMA_LOCK_KEY}
    assert events[2] == ("create_all", True) and events[3] == ("commit",)                  # created on the connection that holds the lock, inside its transaction


def test_other_databases_create_the_schema_without_a_lock(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(db.metadata, "create_all", lambda bind: calls.append(bind))
    registry = Registry.__new__(Registry)
    registry.engine = sa.create_engine(f"sqlite:///{tmp_path / 'x.sqlite3'}", future=True)     # not db.make_engine: on a PostgreSQL test run that would be redirected
    registry._create_schema()
    assert calls == [registry.engine]
