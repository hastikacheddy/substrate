"""The experiment registry: tenants, principals, experiments, runs, tasks, artifacts, datasets and the hash-chained event log.

Every method that touches user data takes the acting `Principal` and filters by its tenant. A lookup of another tenant's object returns None (or an
empty list), exactly as for an object that does not exist, so identifiers cannot be used to probe other tenants.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import sqlalchemy as sa

from ..errors import ValidationError
from . import db
from .security import (DEFAULT_QUOTAS, PER_TENANT_NOT_PER_DAY, Action, Principal, QuotaExceededError, Role, hash_secret,
                       new_api_key, secrets_match, split_token)

GENESIS = "0" * 64
WORKFLOW_VERSION = "1"
COUNTERS = ("tenant", "principal", "api_key", "experiment", "run")
APPEND_ATTEMPTS = 60
SCHEMA_LOCK_KEY = 0x5B57_5254_0001          # the PostgreSQL advisory-lock number that serialises schema creation ("SUBSTRATE" in spirit; any constant would do)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_reject)


def _reject(value):
    raise TypeError(f"{type(value).__name__} is not JSON data")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass
class TaskSpec:
    """What the orchestrator asks the registry to create for a run."""
    name: str
    kind: str                                            # validate | ingest | solve | translate | uncertainty | check | report
    queue: str = "cpu"
    payload: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # names of other tasks in the same run
    max_attempts: int = 3
    timeout_s: float = 600.0
    resources: dict = field(default_factory=dict)


def _row(row) -> dict | None:
    return None if row is None else dict(row._mapping)


class Registry:
    def __init__(self, url: str | sa.engine.Engine = "sqlite:///substrate.sqlite3", *, create: bool = True, clock: Callable[[], float] = time.time):
        self.engine = url if isinstance(url, sa.engine.Engine) else db.make_engine(url)
        self.clock = clock
        if create:
            self._create_schema()
            for name in COUNTERS:
                try:
                    with self.engine.begin() as conn:
                        if conn.execute(sa.select(db.counters.c.name).where(db.counters.c.name == name)).first() is None:
                            conn.execute(sa.insert(db.counters).values(name=name, value=0))
                except sa.exc.IntegrityError:               # another process seeded it at the same moment
                    pass

    def _create_schema(self) -> None:
        """Create the tables that are missing. On PostgreSQL, API and worker pods that start together would otherwise race to create the same tables (one fails with
        "relation already exists" or a duplicate type name), so the creation runs inside a transaction that holds an advisory lock until it commits."""
        if self.engine.dialect.name != "postgresql":
            db.metadata.create_all(self.engine)
            return
        with self.engine.begin() as conn:
            conn.execute(sa.text("SELECT pg_advisory_xact_lock(:key)"), {"key": SCHEMA_LOCK_KEY})
            db.metadata.create_all(conn)

    # -- identifiers ------------------------------------------------------------------------------------------------------------
    def _next(self, conn, name: str) -> int:
        """The next number of the named counter. The UPDATE takes the row's write lock, so the SELECT after it sees this transaction's value
        and no other transaction can take the same number."""
        if name not in COUNTERS:
            raise KeyError(f"unknown counter '{name}'")
        conn.execute(sa.update(db.counters).where(db.counters.c.name == name).values(value=db.counters.c.value + 1))
        return conn.execute(sa.select(db.counters.c.value).where(db.counters.c.name == name)).scalar_one()

    # -- tenants, principals, keys ---------------------------------------------------------------------------------------------
    def create_tenant(self, name: str) -> str:
        with self.engine.begin() as conn:
            tenant_id = f"T-{self._next(conn, 'tenant'):04d}"
            conn.execute(sa.insert(db.tenants).values(id=tenant_id, name=name, created_at=self.clock()))
        return tenant_id

    def create_principal(self, tenant_id: str, name: str, role: Role | str) -> Principal:
        role = Role(role)
        with self.engine.begin() as conn:
            principal_id = f"P-{self._next(conn, 'principal'):04d}"
            conn.execute(sa.insert(db.principals).values(id=principal_id, tenant_id=tenant_id, name=name, role=role.value, disabled=False,
                                                         created_at=self.clock()))
        return Principal(principal_id, tenant_id, name, role)

    def issue_api_key(self, principal_id: str) -> str:
        """A new API key for the principal. The returned token is the only place the secret ever appears."""
        with self.engine.begin() as conn:
            key_id = f"K{self._next(conn, 'api_key'):05d}"
            token, key_hash = new_api_key(key_id)
            conn.execute(sa.insert(db.api_keys).values(id=key_id, principal_id=principal_id, key_hash=key_hash, created_at=self.clock()))
        return token

    def revoke_api_key(self, key_id: str) -> bool:
        with self.engine.begin() as conn:
            done = conn.execute(sa.update(db.api_keys).where(db.api_keys.c.id == key_id, db.api_keys.c.revoked_at.is_(None))
                                .values(revoked_at=self.clock()))
        return done.rowcount == 1

    def authenticate(self, token: str) -> Principal | None:
        """The principal a token belongs to, or None (unknown, malformed, revoked or disabled: the caller cannot tell which)."""
        parts = split_token(token or "")
        if parts is None:
            return None
        key_id, secret = parts
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(db.api_keys.c.key_hash, db.api_keys.c.revoked_at, db.principals.c.id, db.principals.c.tenant_id, db.principals.c.name,
                          db.principals.c.role, db.principals.c.disabled)
                .join(db.principals, db.principals.c.id == db.api_keys.c.principal_id).where(db.api_keys.c.id == key_id)).first()
        stored_hash = row.key_hash if row else hash_secret("no such key")        # compare either way, so a wrong id costs the same as a wrong secret
        matches = secrets_match(secret, stored_hash)
        if row is None or not matches or row.revoked_at is not None or row.disabled:
            return None
        return Principal(row.id, row.tenant_id, row.name, Role(row.role))

    def principal_in_tenant(self, tenant_id: str, principal_id: str) -> Principal | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(db.principals).where(db.principals.c.id == principal_id, db.principals.c.tenant_id == tenant_id,
                                                              db.principals.c.disabled.is_(False))).first()
        return None if row is None else Principal(row.id, row.tenant_id, row.name, Role(row.role))

    def bootstrap(self, tenant_name: str, admin_name: str = "admin") -> tuple[str, Principal, str]:
        """A tenant with one administrator and a key for them: (tenant id, principal, token)."""
        tenant_id = self.create_tenant(tenant_name)
        admin = self.create_principal(tenant_id, admin_name, Role.ADMIN)
        return tenant_id, admin, self.issue_api_key(admin.id)

    # -- the event log (hash chains) --------------------------------------------------------------------------------------------
    def _append_event(self, conn, chain: str, tenant_id: str, kind: str, payload: dict) -> str:
        previous = conn.execute(sa.select(db.events.c.hash).where(db.events.c.chain == chain).order_by(db.events.c.seq.desc()).limit(1)).scalar()
        prev_hash = previous or GENESIS
        ts = self.clock()
        digest = sha256_json([prev_hash, chain, ts, kind, payload])
        conn.execute(sa.insert(db.events).values(chain=chain, tenant_id=tenant_id, ts=ts, kind=kind, payload=payload, prev_hash=prev_hash, hash=digest))
        return digest

    def append_event(self, chain: str, tenant_id: str, kind: str, payload: dict) -> str:
        """Add an event to a chain. The unique (chain, prev_hash) index makes a concurrent writer fail instead of forking the chain, so retry."""
        for attempt in range(APPEND_ATTEMPTS):
            try:
                with self.engine.begin() as conn:
                    return self._append_event(conn, chain, tenant_id, kind, payload)
            except (sa.exc.IntegrityError, sa.exc.OperationalError):     # lost the race for the link (or, on SQLite, for the write lock): look again
                if attempt == APPEND_ATTEMPTS - 1:
                    raise
                time.sleep(random.random() * 0.005 * (attempt + 1))
        raise AssertionError("unreachable")

    def audit(self, tenant_id: str, kind: str, **payload) -> str:
        return self.append_event(f"audit:{tenant_id}", tenant_id, kind, payload)

    def events(self, chain: str, tenant_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.events).where(db.events.c.chain == chain, db.events.c.tenant_id == tenant_id).order_by(db.events.c.seq)).all()
        return [dict(r._mapping) for r in rows]

    def verify_chain(self, chain: str, tenant_id: str) -> tuple[bool, int | None]:
        """(intact, sequence number of the first event that does not match). An edit, a deletion or a reordering all break the chain."""
        expected_prev = GENESIS
        for event in self.events(chain, tenant_id):
            recomputed = sha256_json([event["prev_hash"], chain, event["ts"], event["kind"], event["payload"]])
            if event["prev_hash"] != expected_prev or event["hash"] != recomputed:
                return False, event["seq"]
            expected_prev = event["hash"]
        return True, None

    # -- quotas and usage -------------------------------------------------------------------------------------------------------
    def quota(self, tenant_id: str, metric: str) -> float:
        if metric not in DEFAULT_QUOTAS:
            raise KeyError(f"unknown quota '{metric}' (have: {', '.join(DEFAULT_QUOTAS)})")
        with self.engine.connect() as conn:
            override = conn.execute(sa.select(db.quota_overrides.c.limit).where(db.quota_overrides.c.tenant_id == tenant_id,
                                                                                  db.quota_overrides.c.metric == metric)).scalar()
        return DEFAULT_QUOTAS[metric] if override is None else override

    def set_quota(self, admin: Principal, metric: str, limit: float) -> None:
        admin.require(Action.ADMIN)
        if metric not in DEFAULT_QUOTAS:
            raise KeyError(f"unknown quota '{metric}'")
        if not limit >= 0:
            raise ValidationError(f"a quota must be a non-negative number (got {limit})")
        with self.engine.begin() as conn:
            changed = conn.execute(sa.update(db.quota_overrides).where(db.quota_overrides.c.tenant_id == admin.tenant_id, db.quota_overrides.c.metric == metric)
                                   .values(limit=float(limit)))
            if changed.rowcount == 0:
                conn.execute(sa.insert(db.quota_overrides).values(tenant_id=admin.tenant_id, metric=metric, limit=float(limit)))
            self._append_event(conn, f"audit:{admin.tenant_id}", admin.tenant_id, "quota.set", {"by": admin.id, "metric": metric, "limit": float(limit)})

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def used_today(self, principal: Principal, metric: str) -> float:
        with self.engine.connect() as conn:
            value = conn.execute(sa.select(db.usage.c.amount).where(db.usage.c.tenant_id == principal.tenant_id, db.usage.c.principal_id == principal.id,
                                                                    db.usage.c.day == self._today(), db.usage.c.metric == metric)).scalar()
        return float(value or 0.0)

    def consume(self, principal: Principal, metric: str, amount: float = 1.0) -> float:
        """Use `amount` of a per-day allowance, or raise QuotaExceededError (and write a security event) without using any."""
        limit = self.quota(principal.tenant_id, metric)
        day = self._today()
        with self.engine.begin() as conn:
            current = conn.execute(sa.select(db.usage.c.amount).where(db.usage.c.tenant_id == principal.tenant_id, db.usage.c.principal_id == principal.id,
                                                                      db.usage.c.day == day, db.usage.c.metric == metric)).scalar()
            used = float(current or 0.0)
            if used + amount > limit:
                self._append_event(conn, f"audit:{principal.tenant_id}", principal.tenant_id, "quota.blocked",
                                   {"principal": principal.id, "metric": metric, "requested": amount, "used": used, "limit": limit, "result": "BLOCKED"})
                blocked = QuotaExceededError(f"{principal.name}: {metric} would reach {used + amount:g} of {limit:g} today")
            else:
                blocked = None
                if current is None:
                    conn.execute(sa.insert(db.usage).values(tenant_id=principal.tenant_id, principal_id=principal.id, day=day, metric=metric, amount=amount))
                else:
                    conn.execute(sa.update(db.usage).where(db.usage.c.tenant_id == principal.tenant_id, db.usage.c.principal_id == principal.id,
                                                           db.usage.c.day == day, db.usage.c.metric == metric).values(amount=db.usage.c.amount + amount))
        if blocked:
            raise blocked
        return used + amount

    # -- experiments ------------------------------------------------------------------------------------------------------------
    def add_experiment(self, principal: Principal, name: str, spec: dict, datasets: list[dict] | None = None) -> dict:
        principal.require(Action.EXPERIMENT_CREATE)
        spec_sha = sha256_json(spec)                                         # also refuses anything that is not plain JSON data
        limit = self.quota(principal.tenant_id, "experiments_total")
        with self.engine.begin() as conn:
            count = conn.execute(sa.select(sa.func.count()).select_from(db.experiments).where(db.experiments.c.tenant_id == principal.tenant_id)).scalar_one()
            if count + 1 > limit:
                raise QuotaExceededError(f"{principal.name}: the tenant already holds {count} experiments (limit {limit:g})")
            experiment_id = f"EXP-{self._next(conn, 'experiment'):06d}"
            conn.execute(sa.insert(db.experiments).values(
                id=experiment_id, tenant_id=principal.tenant_id, name=name[:200], spec=json.loads(canonical_json(spec)), spec_sha256=spec_sha,
                datasets=list(datasets or []), workflow_version=WORKFLOW_VERSION, created_by=principal.id, created_at=self.clock()))
            self._append_event(conn, f"audit:{principal.tenant_id}", principal.tenant_id, "experiment.create", {"by": principal.id, "experiment": experiment_id,
                                                                                                                   "spec_sha256": spec_sha})
        return self.get_experiment(principal, experiment_id)

    def get_experiment(self, principal: Principal, experiment_id: str) -> dict | None:
        principal.require(Action.EXPERIMENT_READ)
        return self.experiment_in_tenant(principal.tenant_id, experiment_id)

    def experiment_in_tenant(self, tenant_id: str, experiment_id: str) -> dict | None:
        """For the orchestrator and workers, which act for a tenant rather than for a person."""
        with self.engine.connect() as conn:
            return _row(conn.execute(sa.select(db.experiments).where(db.experiments.c.id == experiment_id,
                                                                     db.experiments.c.tenant_id == tenant_id)).first())

    def list_experiments(self, principal: Principal, limit: int = 50, offset: int = 0) -> list[dict]:
        principal.require(Action.EXPERIMENT_READ)
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.experiments.c.id, db.experiments.c.name, db.experiments.c.spec_sha256, db.experiments.c.created_at,
                                          db.experiments.c.created_by).where(db.experiments.c.tenant_id == principal.tenant_id)
                                .order_by(db.experiments.c.created_at.desc(), db.experiments.c.id.desc()).limit(max(1, min(limit, 500))).offset(max(0, offset))).all()
        return [dict(r._mapping) for r in rows]

    # -- runs and tasks ---------------------------------------------------------------------------------------------------------
    def create_run(self, principal: Principal, experiment_id: str, *, seed: int, n_samples: int, environment: dict, environment_hash: str,
                   tasks: list[TaskSpec], reproduces: str | None = None) -> dict:
        """Create a run and all its tasks in one transaction. Tasks with no dependencies start `ready`, the rest `pending`."""
        principal.require(Action.RUN_REPRODUCE if reproduces else Action.RUN_CREATE)
        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            raise ValidationError("two tasks of a run have the same name")
        for task in tasks:
            if missing := [d for d in task.depends_on if d not in names]:
                raise ValidationError(f"task '{task.name}' depends on unknown task(s): {', '.join(missing)}")
        concurrent_limit = self.quota(principal.tenant_id, "concurrent_runs")
        with self.engine.begin() as conn:
            experiment = conn.execute(sa.select(db.experiments.c.id).where(db.experiments.c.id == experiment_id,
                                                                           db.experiments.c.tenant_id == principal.tenant_id)).first()
            if experiment is None:
                raise LookupError(f"no experiment {experiment_id}")
            active = conn.execute(sa.select(sa.func.count()).select_from(db.runs).where(db.runs.c.tenant_id == principal.tenant_id,
                                                                                         db.runs.c.state.in_(["queued", "running"]))).scalar_one()
            if active + 1 > concurrent_limit:
                self._append_event(conn, f"audit:{principal.tenant_id}", principal.tenant_id, "quota.blocked",
                                   {"principal": principal.id, "metric": "concurrent_runs", "requested": 1, "used": active, "limit": concurrent_limit,
                                    "result": "BLOCKED"})
                blocked = QuotaExceededError(f"{principal.name}: {active} runs are already active (limit {concurrent_limit:g})")
            else:
                blocked = None
            if not blocked:
                run_id = f"RUN-{self._next(conn, 'run'):06d}"
                conn.execute(sa.insert(db.runs).values(
                    id=run_id, experiment_id=experiment_id, tenant_id=principal.tenant_id, seed=seed, n_samples=n_samples, state="queued",
                    environment=environment, environment_hash=environment_hash, reproduces=reproduces, created_by=principal.id, created_at=self.clock()))
                ids = {t.name: f"{run_id}/{i:02d}" for i, t in enumerate(tasks)}
                for task in tasks:
                    conn.execute(sa.insert(db.tasks).values(
                        id=ids[task.name], run_id=run_id, tenant_id=principal.tenant_id, name=task.name, kind=task.kind, queue=task.queue,
                        payload=json.loads(canonical_json(task.payload)), depends_on=[ids[d] for d in task.depends_on],
                        state="pending" if task.depends_on else "ready", attempts=0, max_attempts=task.max_attempts, timeout_s=task.timeout_s,
                        resources=task.resources, not_before=0.0))
                self._append_event(conn, f"audit:{principal.tenant_id}", principal.tenant_id, "run.create",
                                   {"by": principal.id, "run": run_id, "experiment": experiment_id, "reproduces": reproduces})
        if blocked:
            raise blocked
        return self.run_in_tenant(principal.tenant_id, run_id)

    def get_run(self, principal: Principal, run_id: str) -> dict | None:
        principal.require(Action.RUN_READ)
        with self.engine.connect() as conn:
            run = _row(conn.execute(sa.select(db.runs).where(db.runs.c.id == run_id, db.runs.c.tenant_id == principal.tenant_id)).first())
        return run

    def list_runs(self, principal: Principal, experiment_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        principal.require(Action.RUN_READ)
        query = sa.select(db.runs.c.id, db.runs.c.experiment_id, db.runs.c.state, db.runs.c.created_at, db.runs.c.finished_at, db.runs.c.seed,
                          db.runs.c.reproduces).where(db.runs.c.tenant_id == principal.tenant_id)
        if experiment_id:
            query = query.where(db.runs.c.experiment_id == experiment_id)
        with self.engine.connect() as conn:
            rows = conn.execute(query.order_by(db.runs.c.created_at.desc(), db.runs.c.id.desc()).limit(max(1, min(limit, 500))).offset(max(0, offset))).all()
        return [dict(r._mapping) for r in rows]

    def reproductions_of(self, principal: Principal, run_id: str) -> list[dict]:
        """The runs in the caller's tenant that were made to reproduce `run_id`, newest first."""
        principal.require(Action.RUN_READ)
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.runs).where(db.runs.c.tenant_id == principal.tenant_id, db.runs.c.reproduces == run_id)
                                .order_by(db.runs.c.created_at.desc(), db.runs.c.id.desc())).all()
        return [dict(r._mapping) for r in rows]

    def run_tasks(self, principal: Principal, run_id: str) -> list[dict]:
        principal.require(Action.RUN_READ)
        return self.tasks_in_tenant(principal.tenant_id, run_id)

    def tasks_in_tenant(self, tenant_id: str, run_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.tasks).where(db.tasks.c.run_id == run_id, db.tasks.c.tenant_id == tenant_id).order_by(db.tasks.c.id)).all()
        return [dict(r._mapping) for r in rows]

    def run_in_tenant(self, tenant_id: str, run_id: str) -> dict | None:
        """For the orchestrator and workers, which act for a tenant rather than for a person."""
        with self.engine.connect() as conn:
            return _row(conn.execute(sa.select(db.runs).where(db.runs.c.id == run_id, db.runs.c.tenant_id == tenant_id)).first())

    def finish_run(self, tenant_id: str, run_id: str, state: str, status: dict, result_digest: str | None) -> bool:
        """Close a run that is still open. Returns whether this call closed it: when several workers race to be the one that does, exactly one wins."""
        with self.engine.begin() as conn:
            done = conn.execute(sa.update(db.runs).where(db.runs.c.id == run_id, db.runs.c.tenant_id == tenant_id, db.runs.c.state.in_(["queued", "running"]))
                                .values(state=state, status=status, result_digest=result_digest, finished_at=self.clock()))
        return done.rowcount == 1

    def mark_run_started(self, tenant_id: str, run_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa.update(db.runs).where(db.runs.c.id == run_id, db.runs.c.tenant_id == tenant_id, db.runs.c.state == "queued")
                         .values(state="running", started_at=self.clock()))

    # -- accounting after the fact ----------------------------------------------------------------------------------------------
    def record_usage(self, tenant_id: str, principal_id: str, metric: str, amount: float) -> None:
        """Add to a per-day total without checking a limit (the work has already been done); `used_by` is what the next request is checked against."""
        day = self._today()
        with self.engine.begin() as conn:
            changed = conn.execute(sa.update(db.usage).where(db.usage.c.tenant_id == tenant_id, db.usage.c.principal_id == principal_id,
                                                             db.usage.c.day == day, db.usage.c.metric == metric).values(amount=db.usage.c.amount + amount))
            if changed.rowcount == 0:
                conn.execute(sa.insert(db.usage).values(tenant_id=tenant_id, principal_id=principal_id, day=day, metric=metric, amount=amount))

    def used_by(self, tenant_id: str, principal_id: str, metric: str) -> float:
        with self.engine.connect() as conn:
            value = conn.execute(sa.select(db.usage.c.amount).where(db.usage.c.tenant_id == tenant_id, db.usage.c.principal_id == principal_id,
                                                                    db.usage.c.day == self._today(), db.usage.c.metric == metric)).scalar()
        return float(value or 0.0)

    # -- versioned datasets ----------------------------------------------------------------------------------------------------
    def put_dataset(self, dataset_id: str, *, source: str, license: str, schema: dict, records: list, provenance: dict, raw_sha256: str | None = None) -> dict:
        """Store a dataset. If the records are identical to the latest version's, that version is returned; otherwise this becomes the next version.
        A stored version is never changed, so an experiment pinned to one cannot be altered by a later fetch."""
        checksum = sha256_json(records)
        with self.engine.begin() as conn:
            latest = conn.execute(sa.select(db.datasets).where(db.datasets.c.dataset_id == dataset_id).order_by(db.datasets.c.version.desc()).limit(1)).first()
            if latest is not None and latest.checksum == checksum:
                return dict(latest._mapping)
            version = 1 if latest is None else latest.version + 1
            conn.execute(sa.insert(db.datasets).values(
                dataset_id=dataset_id, version=version, source=source, retrieved_at=self.clock(), license=license, schema=schema, checksum=checksum,
                raw_sha256=raw_sha256, records=json.loads(canonical_json(records)), provenance=json.loads(canonical_json(provenance)), created_at=self.clock()))
            return dict(conn.execute(sa.select(db.datasets).where(db.datasets.c.dataset_id == dataset_id, db.datasets.c.version == version)).one()._mapping)

    def get_dataset(self, dataset_id: str, version: int | None = None) -> dict | None:
        query = sa.select(db.datasets).where(db.datasets.c.dataset_id == dataset_id)
        query = query.where(db.datasets.c.version == version) if version is not None else query.order_by(db.datasets.c.version.desc()).limit(1)
        with self.engine.connect() as conn:
            return _row(conn.execute(query).first())

    def list_datasets(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(db.datasets.c.dataset_id, db.datasets.c.version, db.datasets.c.source, db.datasets.c.license,
                                          db.datasets.c.checksum, db.datasets.c.retrieved_at).order_by(db.datasets.c.dataset_id, db.datasets.c.version)).all()
        return [dict(r._mapping) for r in rows]

    # -- artifacts: metadata (the bytes live in an ArtifactStore) -----------------------------------------------------------------
    def record_artifact(self, tenant_id: str, sha256: str, *, run_id: str | None, name: str, media_type: str, size: int) -> None:
        """Record that `sha256` exists for the tenant and (if `run_id`) that the run has it under `name`. Both are idempotent, and safe when several
        workers record the same content at once: whoever loses the race finds the row already there and is done."""
        for attempt in range(4):
            try:
                with self.engine.begin() as conn:
                    if conn.execute(sa.select(db.artifacts.c.sha256).where(db.artifacts.c.tenant_id == tenant_id, db.artifacts.c.sha256 == sha256)).first() is None:
                        conn.execute(sa.insert(db.artifacts).values(tenant_id=tenant_id, sha256=sha256, run_id=run_id, name=name[:200], media_type=media_type,
                                                                    size=size, created_at=self.clock()))
                    if run_id and conn.execute(sa.select(db.artifact_links.c.sha256).where(
                            db.artifact_links.c.tenant_id == tenant_id, db.artifact_links.c.run_id == run_id, db.artifact_links.c.sha256 == sha256,
                            db.artifact_links.c.name == name[:200])).first() is None:
                        conn.execute(sa.insert(db.artifact_links).values(tenant_id=tenant_id, run_id=run_id, sha256=sha256, name=name[:200]))
                return
            except sa.exc.IntegrityError:
                if attempt == 3:
                    raise

    def artifact_meta(self, principal: Principal, sha256: str) -> dict | None:
        principal.require(Action.ARTIFACT_READ)
        return self.artifact_meta_in_tenant(principal.tenant_id, sha256)

    def artifact_meta_in_tenant(self, tenant_id: str, sha256: str) -> dict | None:
        with self.engine.connect() as conn:
            return _row(conn.execute(sa.select(db.artifacts).where(db.artifacts.c.tenant_id == tenant_id, db.artifacts.c.sha256 == sha256)).first())

    def run_artifacts(self, principal: Principal, run_id: str) -> list[dict]:
        """The artifacts a run has, each under the name the run gave it (the same bytes can be in many runs)."""
        principal.require(Action.ARTIFACT_READ)
        links = db.artifact_links
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(db.artifacts.c.tenant_id, db.artifacts.c.sha256, links.c.run_id, links.c.name, db.artifacts.c.media_type, db.artifacts.c.size,
                          db.artifacts.c.created_at)
                .join(links, sa.and_(links.c.tenant_id == db.artifacts.c.tenant_id, links.c.sha256 == db.artifacts.c.sha256))
                .where(links.c.tenant_id == principal.tenant_id, links.c.run_id == run_id).order_by(db.artifacts.c.created_at, links.c.name)).all()
        return [dict(r._mapping) for r in rows]

    def artifact_bytes_total(self, tenant_id: str) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(sa.select(sa.func.coalesce(sa.func.sum(db.artifacts.c.size), 0)).where(db.artifacts.c.tenant_id == tenant_id)).scalar_one())
