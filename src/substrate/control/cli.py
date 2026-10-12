"""`python -m substrate platform ...`: set up a deployment, serve the API, run workers, and drive the control plane from a terminal.

Where things live is given by `--db` / `SUBSTRATE_DB` (a SQLAlchemy URL: SQLite for one machine, PostgreSQL for a deployment) and `--store` / `SUBSTRATE_STORE` (a directory).
Who you are is given by an API key in the environment (`SUBSTRATE_API_KEY`), never on the command line, where it would be visible to every process on the machine.
`init` is the one command that needs no key: it creates the first tenant and administrator and prints that administrator's key, once.
"""
from __future__ import annotations

import os
import sys
import threading
import time

from ..errors import SubstrateError

DEFAULT_DB = "sqlite:///substrate-platform.sqlite3"
DEFAULT_STORE = "substrate-artifacts"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def add_parser(subparsers) -> None:
    platform = subparsers.add_parser("platform", help="the control plane: registry, workers, API (needs `pip install substrate[platform]`)")
    platform.add_argument("--db", default=None, help=f"database URL (default: $SUBSTRATE_DB, else {DEFAULT_DB})")
    platform.add_argument("--store", default=None, help=f"artifact directory (default: $SUBSTRATE_STORE, else {DEFAULT_STORE})")
    sub = platform.add_subparsers(dest="platform_command", required=True)

    init = sub.add_parser("init", help="create the database, a tenant and its administrator; prints the administrator's key once")
    init.add_argument("--tenant", default="default")
    init.add_argument("--admin", default="admin")

    serve = sub.add_parser("serve", help="serve the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--allow-external", action="store_true", help="listen on a non-loopback address (put a TLS-terminating proxy in front)")
    serve.add_argument("--no-metrics", action="store_true")

    worker = sub.add_parser("worker", help="run a worker")
    worker.add_argument("--queues", default="cpu", help="comma-separated queues to serve (cpu, chemistry, quantum, ...)")
    worker.add_argument("--isolation", choices=("inline", "process"), default=None, help="run each step in the worker's process or in its own (default: $SUBSTRATE_WORKER_ISOLATION, else inline)")
    worker.add_argument("--until-idle", action="store_true", help="exit when there is nothing left to take")

    submit = sub.add_parser("submit", help="store an experiment file and start a run of it")
    submit.add_argument("experiment")
    submit.add_argument("--name")
    submit.add_argument("--samples", type=int)
    submit.add_argument("--seed", type=int)
    submit.add_argument("--wait", action="store_true", help="work on the run in this process until it ends, then print its status")

    status = sub.add_parser("status", help="show a run's state and its six-axis status")
    status.add_argument("run")

    reproduce = sub.add_parser("reproduce", help="run an earlier run again and compare the results")
    reproduce.add_argument("run")
    reproduce.add_argument("--wait", action="store_true")

    cancel = sub.add_parser("cancel", help="cancel a run")
    cancel.add_argument("run")

    key = sub.add_parser("apikey", help="manage API keys (administrators)")
    key_sub = key.add_subparsers(dest="key_command", required=True)
    create = key_sub.add_parser("create", help="create a person (or service) and a key for them; prints the key once")
    create.add_argument("name")
    create.add_argument("--role", required=True, choices=("ADMIN", "RESEARCHER", "OPERATOR", "VIEWER", "SERVICE"))
    revoke = key_sub.add_parser("revoke", help="revoke a key by its id (the part of the key between the first two underscores)")
    revoke.add_argument("key_id")


def database_url(explicit: str | None) -> str:
    """The database URL: the argument, else $SUBSTRATE_DB, else the contents of the file named by $SUBSTRATE_DB_FILE (how a container receives it from a secret,
    so the URL, which holds a password, is never in the process environment), else the local SQLite default."""
    if explicit:
        return explicit
    if os.environ.get("SUBSTRATE_DB"):
        return os.environ["SUBSTRATE_DB"]
    path = os.environ.get("SUBSTRATE_DB_FILE")
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                url = handle.read().strip()
        except OSError as error:
            raise SubstrateError(f"cannot read the database URL from {path} ({type(error).__name__})") from None
        if not url:
            raise SubstrateError(f"{path} is empty: it should hold the database URL")
        return url
    return DEFAULT_DB


def artifact_store(location: str | None):
    """A directory (a local store) or `s3://bucket/prefix` (an S3-compatible service: credentials and endpoint come from the standard AWS_* variables)."""
    from .artifacts import LocalArtifactStore, S3ArtifactStore
    location = location or os.environ.get("SUBSTRATE_STORE", DEFAULT_STORE)
    if location.startswith("s3://"):
        bucket, _, prefix = location[5:].partition("/")
        if not bucket:
            raise SubstrateError("an s3 store is written s3://bucket/prefix")
        options = {"endpoint_url": os.environ["SUBSTRATE_S3_ENDPOINT"]} if os.environ.get("SUBSTRATE_S3_ENDPOINT") else {}
        try:
            return S3ArtifactStore.from_boto3(bucket, prefix or "artifacts", **options)
        except ImportError:
            raise SubstrateError("an s3 store needs boto3 (pip install boto3)") from None
    return LocalArtifactStore(location)


def _open(args):
    from .artifacts import Artifacts
    from .registry import Registry
    registry = Registry(database_url(args.db))
    return registry, Artifacts(registry, artifact_store(args.store))


def _principal(registry):
    token = os.environ.get("SUBSTRATE_API_KEY", "")
    principal = registry.authenticate(token)
    if principal is None:
        raise SubstrateError("a valid API key is required in SUBSTRATE_API_KEY (create one with `platform init` or `platform apikey create`)")
    return principal


def _orchestrator(registry, artifacts, isolation: str | None = None, observer=None):
    from .orchestrator import Orchestrator
    from .sandbox import ProcessExecutor, default_isolation
    orchestrator = Orchestrator(registry, artifacts, observer=observer)
    if (isolation or default_isolation()) == "process":
        orchestrator.executor = ProcessExecutor(quantum_policy=orchestrator.gateway.policy)
    return orchestrator


def _print_status(orchestrator, principal, run_id: str) -> int:
    from .status import RunStatus, State
    description = orchestrator.describe(principal, run_id)
    if description is None:
        raise SubstrateError(f"no run {run_id}")
    print(f"{run_id}: {description['state']}  (tasks: {', '.join(f'{n} {s}' for s, n in sorted(description['progress'].items())) or 'none'})")
    if description["status"]:
        status = RunStatus.from_dict(description["status"])
        print(status.summary())
        return 0 if status.overall() in (State.PASS, State.WARN) else 2
    return 0


def create_server_app(registry, artifacts, orchestrator, metrics: bool = True):
    """The API application, with metrics and the bundled data adapters wired in."""
    from .api import create_app
    from .fabric.service import Fabric
    from .observability import MetricsObserver, with_live_gauges
    observer = None
    if metrics:
        observer = with_live_gauges(MetricsObserver(), registry, orchestrator.queue)
        orchestrator.observer = observer
    return create_app(registry, artifacts, orchestrator, metrics=observer, fabric=Fabric(registry, artifacts))


def run(args) -> int:
    try:
        try:
            registry, artifacts = _open(args)
        except ImportError as error:
            raise SubstrateError(f"the control plane needs its extra dependencies: pip install 'substrate[platform]' ({error.name} is missing)") from None
        command = args.platform_command

        if command == "init":
            tenant_id, admin, token = registry.bootstrap(args.tenant, args.admin)
            print(f"tenant {tenant_id} ({args.tenant}) and administrator {admin.id} ({args.admin}) created")
            print("API key (shown once; keep it secret, and export it as SUBSTRATE_API_KEY):")
            print(token)
            return 0

        if command == "serve":
            if args.host not in LOOPBACK and not args.allow_external:
                raise SubstrateError(f"refusing to listen on {args.host}: the API speaks plain HTTP, so it is loopback-only unless --allow-external is given (and a TLS proxy is in front)")
            import uvicorn
            orchestrator = _orchestrator(registry, artifacts)
            uvicorn.run(create_server_app(registry, artifacts, orchestrator, not args.no_metrics), host=args.host, port=args.port, log_level="info", server_header=False)
            return 0

        if command == "worker":
            from .worker import Worker
            from .observability import CompositeObserver, JsonFormatter, LoggingObserver, configure_tracing
            import logging
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
            logging.getLogger("substrate.control").addHandler(handler)
            logging.getLogger("substrate.control").setLevel(logging.INFO)
            try:
                tracing = configure_tracing()
            except (ValueError, ImportError) as error:
                raise SubstrateError(str(error)) from None
            observer = CompositeObserver(LoggingObserver(), tracing) if tracing else LoggingObserver()
            orchestrator = _orchestrator(registry, artifacts, args.isolation, observer)
            worker = Worker(orchestrator, queues=[q.strip() for q in args.queues.split(",") if q.strip()])
            stop = threading.Event()
            try:
                if args.until_idle:
                    print(f"{worker.worker_id}: ran {worker.run_until_idle()} task(s)")
                else:
                    worker.serve(stop)
            except KeyboardInterrupt:
                stop.set()
            finally:
                if tracing:
                    tracing.shutdown()                          # sends the spans still waiting in the exporter's buffer
            return 0

        principal = _principal(registry)
        orchestrator = _orchestrator(registry, artifacts)

        if command == "submit":
            from ..experiment import read_spec
            spec = read_spec(args.experiment)
            orchestrator.check_spec(principal, spec)
            name = args.name or os.path.splitext(os.path.basename(args.experiment))[0]
            experiment = registry.add_experiment(principal, name, spec)
            run_record = orchestrator.submit(principal, experiment["id"], seed=args.seed, n_samples=args.samples)
            print(f"{experiment['id']} -> {run_record['id']}")
            if args.wait:
                _work_until_done(orchestrator, principal, run_record["id"])
                return _print_status(orchestrator, principal, run_record["id"])
            return 0

        if command == "reproduce":
            run_record = orchestrator.reproduce(principal, args.run)
            print(f"{args.run} -> {run_record['id']} (a reproduction)")
            if args.wait:
                _work_until_done(orchestrator, principal, run_record["id"])
                return _print_status(orchestrator, principal, run_record["id"])
            return 0

        if command == "status":
            return _print_status(orchestrator, principal, args.run)

        if command == "cancel":
            print(f"cancelled {orchestrator.cancel(principal, args.run)} task(s) of {args.run}")
            return 0

        if command == "apikey":
            from .security import Action, Role
            principal.require(Action.ADMIN)
            if args.key_command == "create":
                person = registry.create_principal(principal.tenant_id, args.name, Role(args.role))
                print(f"{person.id} ({args.name}, {args.role}); API key (shown once):")
                print(registry.issue_api_key(person.id))
                return 0
            print("revoked" if registry.revoke_api_key(args.key_id) else f"no active key {args.key_id}")
            return 0
        raise SubstrateError(f"unknown command {command}")
    except SubstrateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except LookupError as error:
        print(f"error: {error.args[0] if error.args else 'not found'}", file=sys.stderr)
        return 1


def _work_until_done(orchestrator, principal, run_id: str, timeout_s: float = 3600.0) -> None:
    from .worker import Worker
    worker = Worker(orchestrator, queues=("cpu", "chemistry", "quantum"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        progressed = worker.run_once()
        run = orchestrator.registry.get_run(principal, run_id)
        if run["state"] in ("succeeded", "failed", "cancelled"):
            return
        if not progressed:
            time.sleep(0.5)
    raise SubstrateError(f"{run_id} did not finish within {timeout_s:g} s")


__all__ = ["add_parser", "run", "create_server_app"]
