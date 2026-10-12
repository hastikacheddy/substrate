"""The database schema: SQLAlchemy Core tables that work on SQLite (development, tests, a single machine) and PostgreSQL (a deployment).

Conventions that matter for safety:
  * every row that belongs to a user carries `tenant_id`, and `Registry` never builds a query for such a table without filtering on it
  * identifiers a person reads (EXP-000421, RUN-000007) come from the `counters` table, inside the same transaction that uses them
  * times are POSIX seconds (a float), so no time-zone or driver difference can change what is stored
  * JSON columns hold plain JSON; nothing in this schema stores a serialised Python object
"""
from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()

counters = sa.Table(
    "counters", metadata,
    sa.Column("name", sa.String(64), primary_key=True),
    sa.Column("value", sa.Integer, nullable=False, default=0),
)

tenants = sa.Table(
    "tenants", metadata,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("name", sa.String(120), nullable=False, unique=True),
    sa.Column("created_at", sa.Float, nullable=False),
)

principals = sa.Table(
    "principals", metadata,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("tenant_id", sa.String(32), sa.ForeignKey("tenants.id"), nullable=False),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("role", sa.String(16), nullable=False),
    sa.Column("disabled", sa.Boolean, nullable=False, default=False),
    sa.Column("created_at", sa.Float, nullable=False),
    sa.UniqueConstraint("tenant_id", "name"),
)

api_keys = sa.Table(
    "api_keys", metadata,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("principal_id", sa.String(32), sa.ForeignKey("principals.id"), nullable=False),
    sa.Column("key_hash", sa.String(64), nullable=False, unique=True),       # SHA-256 of the secret; the secret itself is shown once and not kept
    sa.Column("created_at", sa.Float, nullable=False),
    sa.Column("revoked_at", sa.Float),
)

experiments = sa.Table(
    "experiments", metadata,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("tenant_id", sa.String(32), sa.ForeignKey("tenants.id"), nullable=False),
    sa.Column("name", sa.String(200), nullable=False),
    sa.Column("spec", sa.JSON, nullable=False),
    sa.Column("spec_sha256", sa.String(64), nullable=False),
    sa.Column("datasets", sa.JSON, nullable=False),                         # [{"id": ..., "version": ...}] the experiment is pinned to
    sa.Column("workflow_version", sa.String(32), nullable=False),
    sa.Column("created_by", sa.String(32), nullable=False),
    sa.Column("created_at", sa.Float, nullable=False),
    sa.Index("ix_experiments_tenant", "tenant_id", "created_at"),
)

runs = sa.Table(
    "runs", metadata,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("experiment_id", sa.String(32), sa.ForeignKey("experiments.id"), nullable=False),
    sa.Column("tenant_id", sa.String(32), sa.ForeignKey("tenants.id"), nullable=False),
    sa.Column("seed", sa.Integer, nullable=False),
    sa.Column("n_samples", sa.Integer, nullable=False),
    sa.Column("state", sa.String(16), nullable=False),                      # queued | running | succeeded | failed | cancelled
    sa.Column("status", sa.JSON),                                           # the six-axis RunStatus, once the run has ended
    sa.Column("environment", sa.JSON, nullable=False),
    sa.Column("environment_hash", sa.String(64), nullable=False),
    sa.Column("result_digest", sa.String(64)),                              # hash of the rounded final observables: what a reproduction must match
    sa.Column("reproduces", sa.String(32)),                                 # the run this one re-executes, if it is a reproduction
    sa.Column("created_by", sa.String(32), nullable=False),
    sa.Column("created_at", sa.Float, nullable=False),
    sa.Column("started_at", sa.Float),
    sa.Column("finished_at", sa.Float),
    sa.Index("ix_runs_tenant", "tenant_id", "created_at"),
    sa.Index("ix_runs_experiment", "experiment_id"),
)

tasks = sa.Table(
    "tasks", metadata,
    sa.Column("id", sa.String(48), primary_key=True),                       # RUN-000007/03
    sa.Column("run_id", sa.String(32), sa.ForeignKey("runs.id"), nullable=False),
    sa.Column("tenant_id", sa.String(32), sa.ForeignKey("tenants.id"), nullable=False),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("kind", sa.String(24), nullable=False),                       # what the task is (validate, solve, ...): decides its failure axis
    sa.Column("queue", sa.String(24), nullable=False),                      # which workers may take it (cpu, chemistry, quantum, gpu, hpc)
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("depends_on", sa.JSON, nullable=False),
    sa.Column("state", sa.String(16), nullable=False),                      # pending | ready | running | succeeded | failed | skipped | cancelled
    sa.Column("attempts", sa.Integer, nullable=False, default=0),
    sa.Column("max_attempts", sa.Integer, nullable=False, default=3),
    sa.Column("timeout_s", sa.Float, nullable=False),
    sa.Column("resources", sa.JSON, nullable=False),
    sa.Column("not_before", sa.Float, nullable=False, default=0.0),         # a retry waits for its backoff
    sa.Column("lease_owner", sa.String(64)),
    sa.Column("lease_expires", sa.Float),
    sa.Column("started_at", sa.Float),
    sa.Column("finished_at", sa.Float),
    sa.Column("error", sa.JSON),
    sa.Column("output", sa.String(64)),                                     # sha256 of the output artifact
    sa.Column("seconds", sa.Float),
    sa.Index("ix_tasks_run", "run_id"),
    sa.Index("ix_tasks_claim", "state", "queue", "not_before"),
)

artifacts = sa.Table(
    "artifacts", metadata,
    sa.Column("tenant_id", sa.String(32), sa.ForeignKey("tenants.id"), primary_key=True),
    sa.Column("sha256", sa.String(64), primary_key=True),
    sa.Column("run_id", sa.String(32)),
    sa.Column("name", sa.String(200), nullable=False),
    sa.Column("media_type", sa.String(80), nullable=False),
    sa.Column("size", sa.Integer, nullable=False),
    sa.Column("created_at", sa.Float, nullable=False),
)

artifact_links = sa.Table(
    "artifact_links", metadata,                                              # which runs produced or used which artifact: the same bytes can belong to many runs
    sa.Column("tenant_id", sa.String(32), primary_key=True),
    sa.Column("run_id", sa.String(32), primary_key=True),
    sa.Column("sha256", sa.String(64), primary_key=True),
    sa.Column("name", sa.String(200), primary_key=True),
    sa.Index("ix_artifact_links_run", "run_id"),
)

datasets = sa.Table(
    "datasets", metadata,
    sa.Column("dataset_id", sa.String(80), primary_key=True),
    sa.Column("version", sa.Integer, primary_key=True),
    sa.Column("source", sa.String(300), nullable=False),
    sa.Column("retrieved_at", sa.Float, nullable=False),
    sa.Column("license", sa.String(200), nullable=False),
    sa.Column("schema", sa.JSON, nullable=False),
    sa.Column("checksum", sa.String(64), nullable=False),                   # of the normalised records
    sa.Column("raw_sha256", sa.String(64)),                                 # of the bytes as received, kept as an artifact
    sa.Column("records", sa.JSON, nullable=False),
    sa.Column("provenance", sa.JSON, nullable=False),
    sa.Column("created_at", sa.Float, nullable=False),
)

events = sa.Table(
    "events", metadata,
    sa.Column("seq", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("chain", sa.String(64), nullable=False),                      # "audit:<tenant>" or "lineage:<run>": each chain is its own hash chain
    sa.Column("tenant_id", sa.String(32), nullable=False),
    sa.Column("ts", sa.Float, nullable=False),
    sa.Column("kind", sa.String(40), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("prev_hash", sa.String(64), nullable=False),
    sa.Column("hash", sa.String(64), nullable=False),
    sa.Index("ix_events_chain", "chain", "seq"),
    sa.UniqueConstraint("chain", "prev_hash"),                            # two writers cannot both extend the same link: the second fails and retries
)

usage = sa.Table(
    "usage", metadata,
    sa.Column("tenant_id", sa.String(32), primary_key=True),
    sa.Column("principal_id", sa.String(32), primary_key=True),
    sa.Column("day", sa.String(10), primary_key=True),                      # UTC date, YYYY-MM-DD
    sa.Column("metric", sa.String(40), primary_key=True),
    sa.Column("amount", sa.Float, nullable=False, default=0.0),
)

quota_overrides = sa.Table(
    "quota_overrides", metadata,
    sa.Column("tenant_id", sa.String(32), primary_key=True),
    sa.Column("metric", sa.String(40), primary_key=True),
    sa.Column("limit", sa.Float, nullable=False),
)


def make_engine(url: str = "sqlite:///substrate.sqlite3", **options):
    """An engine for `url`. SQLite is opened so that the API's threads can share it and foreign keys are enforced."""
    kwargs = dict(options)
    if url.startswith("sqlite"):
        kwargs.setdefault("connect_args", {"check_same_thread": False, "timeout": 60})
    engine = sa.create_engine(url, future=True, **kwargs)
    if url.startswith("sqlite"):
        @sa.event.listens_for(engine, "connect")
        def _pragmas(connection, _record):                    # noqa: ANN001 (SQLAlchemy callback)
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine
