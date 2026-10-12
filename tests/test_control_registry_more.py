"""Cases of the registry that the first tests did not pin, found by mutation testing: lookups that must stay inside a tenant, allowances that are per person, an audit record that must say
what was set, and a hash chain that has to survive losing a race for the link."""
import pytest

sa = pytest.importorskip("sqlalchemy")

from substrate.control import db  # noqa: E402
from substrate.control.registry import APPEND_ATTEMPTS, Registry, TaskSpec  # noqa: E402
from substrate.control.security import Action, QuotaExceededError, Role  # noqa: E402

from test_control_registry import SPEC, Clock  # noqa: E402


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry(tmp_path, clock):
    return Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}", clock=clock)


@pytest.fixture
def two(registry):
    """Two tenants, each with an administrator and a researcher."""
    world = {}
    for name in ("alpha", "beta"):
        tenant, admin, _ = registry.bootstrap(name)
        world[name] = {"tenant": tenant, "admin": admin, "ada": registry.create_principal(tenant, f"ada-{name}", Role.RESEARCHER),
                       "bob": registry.create_principal(tenant, f"bob-{name}", Role.RESEARCHER)}
    return world


# -- principals ----------------------------------------------------------------------------------------------------------------------
def test_a_principal_is_looked_up_only_in_its_own_tenant_and_only_while_enabled(registry, two):
    ada = two["alpha"]["ada"]
    assert registry.principal_in_tenant(two["alpha"]["tenant"], ada.id) == ada
    assert registry.principal_in_tenant(two["beta"]["tenant"], ada.id) is None                  # someone else's person is not found
    with registry.engine.begin() as conn:
        conn.execute(sa.update(db.principals).where(db.principals.c.id == ada.id).values(disabled=True))
    assert registry.principal_in_tenant(two["alpha"]["tenant"], ada.id) is None                 # and a disabled one is not found at all


# -- allowances are per person and per tenant -----------------------------------------------------------------------------------------
def test_an_allowance_used_by_one_person_is_not_used_by_another(registry, two):
    ada, bob = two["alpha"]["ada"], two["alpha"]["bob"]
    registry.consume(ada, "runs_per_day", 3)
    assert registry.used_today(ada, "runs_per_day") == 3 and registry.used_today(bob, "runs_per_day") == 0
    registry.record_usage(two["alpha"]["tenant"], ada.id, "compute_seconds_per_day", 7.5)
    assert registry.used_by(two["alpha"]["tenant"], ada.id, "compute_seconds_per_day") == 7.5
    assert registry.used_by(two["alpha"]["tenant"], bob.id, "compute_seconds_per_day") == 0
    assert registry.used_by(two["beta"]["tenant"], ada.id, "compute_seconds_per_day") == 0       # the same id in another tenant is another row


def test_usage_adds_up_whether_it_is_consumed_or_recorded_afterwards(registry, two):
    ada, tenant = two["alpha"]["ada"], two["alpha"]["tenant"]
    registry.consume(ada, "tasks_per_day", 2)
    registry.consume(ada, "tasks_per_day", 3)
    assert registry.used_today(ada, "tasks_per_day") == 5
    registry.record_usage(tenant, ada.id, "tasks_per_day", 4)
    registry.record_usage(tenant, ada.id, "tasks_per_day", 1.5)
    assert registry.used_today(ada, "tasks_per_day") == 10.5
    registry.record_usage(tenant, ada.id, "brand_new_metric", 2)                                 # the first record of a metric on a day is stored, not lost
    assert registry.used_by(tenant, ada.id, "brand_new_metric") == 2


def test_a_quota_override_belongs_to_the_tenant_that_set_it_and_is_audited_with_its_value(registry, two):
    alpha, beta = two["alpha"], two["beta"]
    registry.set_quota(beta["admin"], "runs_per_day", 7)                                          # beta already has an override when alpha sets its own
    registry.set_quota(alpha["admin"], "runs_per_day", 3)
    registry.set_quota(alpha["admin"], "runs_per_day", 4)                                         # setting it again updates it
    assert registry.quota(alpha["tenant"], "runs_per_day") == 4 and registry.quota(beta["tenant"], "runs_per_day") == 7
    other = registry.quota(two["alpha"]["tenant"], "tasks_per_day")
    assert other == registry.quota(two["beta"]["tenant"], "tasks_per_day")                       # a metric nobody overrode is the default everywhere
    sets = [e for e in registry.events(f"audit:{alpha['tenant']}", alpha["tenant"]) if e["kind"] == "quota.set"]
    assert [e["payload"] for e in sets] == [{"by": alpha["admin"].id, "metric": "runs_per_day", "limit": 3.0}, {"by": alpha["admin"].id, "metric": "runs_per_day", "limit": 4.0}]


def test_the_experiment_quota_counts_only_the_tenants_own_experiments(registry, two):
    alpha, beta = two["alpha"], two["beta"]
    registry.set_quota(alpha["admin"], "experiments_total", 2)
    for i in range(5):
        registry.add_experiment(beta["ada"], f"b{i}", SPEC)                                       # five in the other tenant
    registry.add_experiment(alpha["ada"], "a0", SPEC)
    registry.add_experiment(alpha["ada"], "a1", SPEC)
    with pytest.raises(QuotaExceededError, match="holds 2 experiments"):
        registry.add_experiment(alpha["ada"], "a2", SPEC)


def test_a_blocked_allowance_is_audited_with_the_figures(registry, two):
    ada = two["alpha"]["ada"]
    registry.set_quota(two["alpha"]["admin"], "runs_per_day", 2)
    registry.consume(ada, "runs_per_day", 2)
    with pytest.raises(QuotaExceededError):
        registry.consume(ada, "runs_per_day", 1)
    blocked = [e for e in registry.events(f"audit:{two['alpha']['tenant']}", two["alpha"]["tenant"]) if e["kind"] == "quota.blocked"]
    assert len(blocked) == 1 and blocked[0]["payload"] == {"principal": ada.id, "metric": "runs_per_day", "requested": 1, "used": 2.0, "limit": 2.0, "result": "BLOCKED"}


# -- the hash chain and a lost race --------------------------------------------------------------------------------------------------
class Flaky:
    """Wraps `Registry._append_event` so it fails a given number of times with a given error before working."""

    def __init__(self, registry, error, failures):
        self.real, self.error, self.failures, self.calls = registry._append_event, error, failures, 0
        registry._append_event = self

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self.real(*args, **kwargs)


@pytest.mark.parametrize("error", [sa.exc.IntegrityError("insert", {}, Exception("another writer took the link")), sa.exc.OperationalError("insert", {}, Exception("database is locked"))])
def test_losing_the_race_for_the_link_or_the_write_lock_is_retried(registry, two, error, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    flaky = Flaky(registry, error, failures=3)
    digest = registry.append_event("audit:x", two["alpha"]["tenant"], "test", {"n": 1})
    assert len(digest) == 64 and flaky.calls == 4
    assert registry.verify_chain("audit:x", two["alpha"]["tenant"]) == (True, None)


def test_a_writer_that_keeps_losing_eventually_gives_up_with_the_error(registry, two, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    flaky = Flaky(registry, sa.exc.IntegrityError("insert", {}, Exception("always taken")), failures=10**6)
    with pytest.raises(sa.exc.IntegrityError):
        registry.append_event("audit:x", two["alpha"]["tenant"], "test", {})
    assert flaky.calls == APPEND_ATTEMPTS


def test_an_error_that_is_not_a_lost_race_is_not_swallowed(registry, two):
    Flaky(registry, ZeroDivisionError("a bug"), failures=1)
    with pytest.raises(ZeroDivisionError):
        registry.append_event("audit:x", two["alpha"]["tenant"], "test", {})


def test_events_are_read_back_oldest_first_and_only_for_their_own_tenant(registry, two):
    alpha = two["alpha"]["tenant"]
    for n in range(3):
        registry.append_event("shared-name", alpha, "e", {"n": n})
    registry.append_event("shared-name", two["beta"]["tenant"], "e", {"n": 99})                   # the same chain name, written by another tenant
    assert [e["payload"]["n"] for e in registry.events("shared-name", alpha)] == [0, 1, 2]
    assert [e["payload"]["n"] for e in registry.events("shared-name", two["beta"]["tenant"])] == [99]


def test_a_missing_ref_on_the_chain_is_the_genesis_and_each_event_links_to_the_one_before(registry, two):
    from substrate.control.registry import GENESIS
    tenant = two["alpha"]["tenant"]
    first = registry.append_event("c", tenant, "a", {})
    second = registry.append_event("c", tenant, "b", {})
    third = registry.append_event("c", tenant, "c", {})
    events = registry.events("c", tenant)
    assert [e["prev_hash"] for e in events] == [GENESIS, first, second] and [e["hash"] for e in events] == [first, second, third]


# -- runs, tasks, artifacts, datasets stay inside their tenant and keep their order -----------------------------------------------------------
def make_run(registry, who, name="x", tasks=None):
    experiment = registry.add_experiment(who, name, SPEC)
    return registry.create_run(who, experiment["id"], seed=1, n_samples=0, environment={}, environment_hash="h",
                               tasks=tasks or [TaskSpec("validate", "validate"), TaskSpec("solve", "solve", depends_on=["validate"])])


def test_runs_tasks_and_their_listings_never_cross_tenants(registry, two, clock):
    alpha, beta = two["alpha"], two["beta"]
    registry.set_quota(alpha["admin"], "concurrent_runs", 50)
    run = make_run(registry, alpha["ada"])
    clock.t += 5
    later = make_run(registry, alpha["ada"], "y")
    assert registry.get_run(beta["ada"], run["id"]) is None and registry.run_in_tenant(beta["tenant"], run["id"]) is None
    assert registry.tasks_in_tenant(beta["tenant"], run["id"]) == [] and registry.run_tasks(beta["ada"], run["id"]) == []
    assert [t["name"] for t in registry.tasks_in_tenant(alpha["tenant"], run["id"])] == ["validate", "solve"]          # in the order they were planned
    assert [r["id"] for r in registry.list_runs(alpha["ada"])] == [later["id"], run["id"]]                            # newest first
    assert registry.list_runs(beta["ada"]) == []
    assert [r["id"] for r in registry.list_runs(alpha["ada"], experiment_id=later["experiment_id"])] == [later["id"]]
    assert registry.list_runs(alpha["ada"], limit=0)[0]["id"] == later["id"] and len(registry.list_runs(alpha["ada"], limit=0)) == 1       # a limit below 1 is read as 1
    assert [r["id"] for r in registry.list_runs(alpha["ada"], limit=1, offset=1)] == [run["id"]] and registry.list_runs(alpha["ada"], offset=-5)[0]["id"] == later["id"]
    assert registry.list_runs(alpha["ada"], limit=10**6) == registry.list_runs(alpha["ada"], limit=500)


def test_a_run_cannot_be_made_of_another_tenants_experiment(registry, two):
    experiment = registry.add_experiment(two["beta"]["ada"], "theirs", SPEC)
    with pytest.raises(LookupError):
        registry.create_run(two["alpha"]["ada"], experiment["id"], seed=1, n_samples=0, environment={}, environment_hash="h", tasks=[TaskSpec("validate", "validate")])


def test_the_concurrent_run_quota_counts_this_tenants_active_runs_and_nobody_elses(registry, two):
    alpha, beta = two["alpha"], two["beta"]
    registry.set_quota(alpha["admin"], "concurrent_runs", 2)
    registry.set_quota(beta["admin"], "concurrent_runs", 50)
    for _ in range(5):
        make_run(registry, beta["ada"])                                                          # beta's runs do not count against alpha
    first, second = make_run(registry, alpha["ada"]), make_run(registry, alpha["ada"])
    with pytest.raises(QuotaExceededError, match="2 runs are already active"):
        make_run(registry, alpha["ada"])
    registry.mark_run_started(alpha["tenant"], second["id"])                                     # a running run is still active
    with pytest.raises(QuotaExceededError):
        make_run(registry, alpha["ada"])
    assert registry.finish_run(alpha["tenant"], first["id"], "succeeded", {}, "d") is True
    make_run(registry, alpha["ada"])                                                             # a slot is free again
    blocked = [e for e in registry.events(f"audit:{alpha['tenant']}", alpha["tenant"]) if e["kind"] == "quota.blocked"]
    assert blocked and blocked[0]["payload"]["metric"] == "concurrent_runs" and blocked[0]["payload"]["limit"] == 2.0 and blocked[0]["payload"]["used"] == 2


def test_only_tasks_without_dependencies_start_ready_and_the_dependencies_are_stored(registry, two):
    run = make_run(registry, two["alpha"]["ada"], tasks=[TaskSpec("a", "validate"), TaskSpec("b", "solve", depends_on=["a"]), TaskSpec("c", "solve", depends_on=["a", "b"])])
    rows = {t["name"]: t for t in registry.tasks_in_tenant(two["alpha"]["tenant"], run["id"])}
    assert [rows[n]["state"] for n in "abc"] == ["ready", "pending", "pending"]
    assert rows["c"]["depends_on"] == [rows["a"]["id"], rows["b"]["id"]] and rows["a"]["depends_on"] == []
    created = [e for e in registry.events(f"audit:{two['alpha']['tenant']}", two["alpha"]["tenant"]) if e["kind"] == "run.create"]
    assert created[-1]["payload"] == {"by": two["alpha"]["ada"].id, "run": run["id"], "experiment": run["experiment_id"], "reproduces": None}


def test_a_run_is_closed_once_by_its_own_tenant_and_started_only_while_queued(registry, two):
    alpha, beta = two["alpha"], two["beta"]
    run = make_run(registry, alpha["ada"])
    assert registry.finish_run(beta["tenant"], run["id"], "failed", {}, None) is False           # not another tenant's to close
    assert registry.get_run(alpha["ada"], run["id"])["state"] == "queued"
    registry.mark_run_started(alpha["tenant"], run["id"])
    started = registry.get_run(alpha["ada"], run["id"])["started_at"]
    assert registry.finish_run(alpha["tenant"], run["id"], "succeeded", {"x": 1}, "digest") is True
    assert registry.finish_run(alpha["tenant"], run["id"], "failed", {"x": 2}, "other") is False  # a closed run stays as it was closed
    done = registry.get_run(alpha["ada"], run["id"])
    assert (done["state"], done["status"], done["result_digest"]) == ("succeeded", {"x": 1}, "digest")
    registry.mark_run_started(alpha["tenant"], run["id"])
    assert registry.get_run(alpha["ada"], run["id"])["state"] == "succeeded" and registry.get_run(alpha["ada"], run["id"])["started_at"] == started


def test_reproductions_of_a_run_are_found_in_its_own_tenant_only(registry, two, clock):
    alpha, beta = two["alpha"], two["beta"]
    registry.set_quota(alpha["admin"], "concurrent_runs", 50)
    registry.set_quota(beta["admin"], "concurrent_runs", 50)
    original = make_run(registry, alpha["ada"])
    experiment = registry.add_experiment(alpha["ada"], "again", SPEC)["id"]
    mine = registry.create_run(alpha["ada"], experiment, seed=1, n_samples=0, environment={}, environment_hash="h", tasks=[TaskSpec("v", "validate")], reproduces=original["id"])
    unrelated = make_run(registry, alpha["ada"], "unrelated")
    theirs_experiment = registry.add_experiment(beta["ada"], "theirs", SPEC)["id"]
    registry.create_run(beta["ada"], theirs_experiment, seed=1, n_samples=0, environment={}, environment_hash="h", tasks=[TaskSpec("v", "validate")], reproduces=original["id"])
    assert [r["id"] for r in registry.reproductions_of(alpha["ada"], original["id"])] == [mine["id"]] and unrelated["id"] not in [r["id"] for r in registry.reproductions_of(alpha["ada"], original["id"])]
    assert [r["reproduces"] for r in registry.reproductions_of(beta["ada"], original["id"])] == [original["id"]] and len(registry.reproductions_of(beta["ada"], original["id"])) == 1


def test_datasets_are_versioned_in_order_and_an_unchanged_fetch_is_the_same_version(registry):
    def put(records):
        return registry.put_dataset("pubchem.x", source="s", license="l", schema={}, records=records, provenance={"n": len(records)})
    v1, again, v2, v3 = put([{"a": 1}]), put([{"a": 1}]), put([{"a": 2}]), put([{"a": 3}])
    assert (v1["version"], again["version"], v2["version"], v3["version"]) == (1, 1, 2, 3) and again["checksum"] == v1["checksum"] != v2["checksum"]
    assert registry.get_dataset("pubchem.x")["version"] == 3 and registry.get_dataset("pubchem.x", 1)["records"] == [{"a": 1}] and registry.get_dataset("pubchem.x", 9) is None
    assert registry.get_dataset("nothing") is None
    registry.put_dataset("a.first", source="s", license="l", schema={}, records=[1], provenance={})
    assert [(d["dataset_id"], d["version"]) for d in registry.list_datasets()] == [("a.first", 1), ("pubchem.x", 1), ("pubchem.x", 2), ("pubchem.x", 3)]
    again_after_change = put([{"a": 3}])                                                          # identical to the latest again: still the latest version, not a fourth
    assert again_after_change["version"] == 3
    put([{"a": 1}])                                                                               # data that went back to an old value is a new version, not the old one
    assert registry.get_dataset("pubchem.x")["version"] == 4


def test_artifacts_are_recorded_once_per_tenant_linked_to_runs_and_counted_per_tenant(registry, two):
    alpha, beta = two["alpha"], two["beta"]
    run = make_run(registry, alpha["ada"])
    other = make_run(registry, beta["ada"])
    sha = "a" * 64
    long_name = "n" * 300
    registry.record_artifact(alpha["tenant"], sha, run_id=run["id"], name=long_name, media_type="text/plain", size=10)
    registry.record_artifact(alpha["tenant"], sha, run_id=run["id"], name=long_name, media_type="text/plain", size=10)         # idempotent
    registry.record_artifact(alpha["tenant"], sha, run_id=None, name="loose", media_type="text/plain", size=10)                # the bytes exist without belonging to a run
    registry.record_artifact(beta["tenant"], "b" * 64, run_id=other["id"], name="theirs", media_type="text/plain", size=1000)
    assert registry.artifact_meta(alpha["ada"], sha)["name"] == "n" * 200 and registry.artifact_meta(beta["ada"], sha) is None
    assert registry.artifact_bytes_total(alpha["tenant"]) == 10 and registry.artifact_bytes_total(beta["tenant"]) == 1000
    assert [a["name"] for a in registry.run_artifacts(alpha["ada"], run["id"])] == ["n" * 200]                           # one link, name cut to 200, no link to run None
    assert registry.run_artifacts(beta["ada"], run["id"]) == [] and registry.run_artifacts(alpha["ada"], other["id"]) == []
    assert registry.artifact_bytes_total("T-0000") == 0


def test_the_recorded_artifact_survives_a_lost_race_to_insert_it(registry, two):
    run = make_run(registry, two["alpha"]["ada"])
    real_begin, state = registry.engine.begin, {"calls": 0}

    class Loser:
        def __init__(self):
            self.context = real_begin()

        def __enter__(self):
            state["calls"] += 1
            if state["calls"] < 3:
                raise sa.exc.IntegrityError("insert", {}, Exception("another worker recorded it first"))
            return self.context.__enter__()

        def __exit__(self, *exc):
            return self.context.__exit__(*exc)

    from unittest import mock
    with mock.patch.object(type(registry.engine), "begin", side_effect=lambda *a, **k: Loser()):
        registry.record_artifact(two["alpha"]["tenant"], "c" * 64, run_id=run["id"], name="n", media_type="text/plain", size=3)
    assert state["calls"] == 3 and registry.artifact_meta(two["alpha"]["ada"], "c" * 64) is not None


# -- the registry itself refuses a role that may not read, not only the API in front of it ---------------------------------------------------
def test_reading_runs_tasks_reproductions_and_artifacts_needs_the_matching_permission_in_the_registry_itself(registry, two):
    from substrate.control.security import AuthorizationError, PERMISSIONS
    alpha = two["alpha"]
    run = make_run(registry, alpha["ada"])
    service = registry.create_principal(alpha["tenant"], "worker", Role.SERVICE)
    operator = registry.create_principal(alpha["tenant"], "olga", Role.OPERATOR)
    viewer = registry.create_principal(alpha["tenant"], "vera", Role.VIEWER)
    readers = {
        "get_run": (Action.RUN_READ, lambda p: registry.get_run(p, run["id"])),
        "run_tasks": (Action.RUN_READ, lambda p: registry.run_tasks(p, run["id"])),
        "reproductions_of": (Action.RUN_READ, lambda p: registry.reproductions_of(p, run["id"])),
        "run_artifacts": (Action.ARTIFACT_READ, lambda p: registry.run_artifacts(p, run["id"])),
        "artifact_meta": (Action.ARTIFACT_READ, lambda p: registry.artifact_meta(p, "a" * 64)),
    }
    for name, (action, call) in readers.items():
        for person in (alpha["admin"], alpha["ada"], operator, viewer, service):
            if action in PERMISSIONS[person.role]:
                call(person)                                                              # allowed: no error
            else:
                with pytest.raises(AuthorizationError):
                    call(person)
    with pytest.raises(AuthorizationError):
        registry.run_artifacts(operator, run["id"])                                       # an operator watches runs but never reads the data


def test_an_experiment_name_is_kept_to_200_characters(registry, two):
    stored = registry.add_experiment(two["alpha"]["ada"], "n" * 300, SPEC)
    assert stored["name"] == "n" * 200
    assert registry.add_experiment(two["alpha"]["ada"], "m" * 200, SPEC)["name"] == "m" * 200
