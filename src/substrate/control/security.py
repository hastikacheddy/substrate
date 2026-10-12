"""Who may do what: principals, roles, per-resource permissions, API keys and quotas.

This is authorisation inside one deployment. Authentication here is by API key (a long random secret, stored only as its SHA-256 hash and shown once);
a deployment that wants single sign-on would put an OIDC-validating proxy in front and map its claims to a `Principal`, which is why nothing below
cares how a principal was established. Every query the registry makes for user data is filtered by the principal's `tenant_id`, so one tenant cannot
see, or even learn the existence of, another's experiments, runs or artifacts.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from enum import Enum

from ..errors import ResourceLimitError, SubstrateError


class AuthorizationError(SubstrateError):
    """The principal's role does not allow the action."""


class QuotaExceededError(ResourceLimitError):
    """The principal's allowance for the period is used up. A kind of resource limit, so an ensemble cannot drop it and carry on."""


class Role(str, Enum):
    ADMIN = "ADMIN"                  # everything, within the tenant
    RESEARCHER = "RESEARCHER"        # define, run, reproduce and read experiments
    OPERATOR = "OPERATOR"            # watch and cancel runs, read metrics and the audit trail; never the data itself
    VIEWER = "VIEWER"                # read experiments, runs and results
    SERVICE = "SERVICE"              # a worker: take tasks and write their outputs


class Action(str, Enum):
    EXPERIMENT_READ = "experiment.read"
    EXPERIMENT_CREATE = "experiment.create"
    RUN_READ = "run.read"
    RUN_CREATE = "run.create"
    RUN_CANCEL = "run.cancel"
    RUN_REPRODUCE = "run.reproduce"
    ARTIFACT_READ = "artifact.read"
    DATASET_READ = "dataset.read"
    DATASET_INGEST = "dataset.ingest"
    TRANSFER_RUN = "transfer.run"
    QPU_USE = "qpu.use"
    QUEUE_WORK = "queue.work"
    METRICS_READ = "metrics.read"
    AUDIT_READ = "audit.read"
    ADMIN = "admin"


_A = Action
PERMISSIONS: dict[Role, frozenset[Action]] = {
    Role.ADMIN: frozenset(Action),
    Role.RESEARCHER: frozenset({_A.EXPERIMENT_READ, _A.EXPERIMENT_CREATE, _A.RUN_READ, _A.RUN_CREATE, _A.RUN_CANCEL, _A.RUN_REPRODUCE,
                                _A.ARTIFACT_READ, _A.DATASET_READ, _A.DATASET_INGEST, _A.TRANSFER_RUN, _A.QPU_USE}),
    Role.OPERATOR: frozenset({_A.RUN_READ, _A.RUN_CANCEL, _A.DATASET_READ, _A.METRICS_READ, _A.AUDIT_READ}),
    Role.VIEWER: frozenset({_A.EXPERIMENT_READ, _A.RUN_READ, _A.ARTIFACT_READ, _A.DATASET_READ}),
    Role.SERVICE: frozenset({_A.QUEUE_WORK}),                  # a worker credential does one thing; it reads no data over the API
}


@dataclass(frozen=True)
class Principal:
    id: str
    tenant_id: str
    name: str
    role: Role

    def can(self, action: Action) -> bool:
        return action in PERMISSIONS[self.role]

    def require(self, action: Action) -> None:
        if not self.can(action):
            raise AuthorizationError(f"{self.name} ({self.role.value}) may not {action.value}")


# -- quotas -----------------------------------------------------------------------------------------------------------------------
#: per principal per UTC day, unless noted; a tenant administrator can override any of these (`Registry.set_quota`)
DEFAULT_QUOTAS: dict[str, float] = {
    "runs_per_day": 200,
    "tasks_per_day": 5_000,
    "compute_seconds_per_day": 32 * 3600,      # 32 CPU-hours
    "qpu_jobs_per_day": 0,                     # off until a tenant administrator grants some
    "concurrent_runs": 4,                      # per tenant, not per day
    "artifact_bytes_total": 50 * 1024**3,      # per tenant, not per day
    "dataset_ingests_per_day": 50,             # fetches from outside sources
    "experiments_total": 10_000,               # per tenant, not per day
}
PER_TENANT_NOT_PER_DAY = {"concurrent_runs", "artifact_bytes_total", "experiments_total"}


# -- API keys ---------------------------------------------------------------------------------------------------------------------
KEY_PREFIX = "sbs"


def new_api_key(key_id: str) -> tuple[str, str]:
    """(token, hash): the token is shown to the user once; only the hash of its secret part is stored."""
    secret = secrets.token_urlsafe(32)
    return f"{KEY_PREFIX}_{key_id}_{secret}", hash_secret(secret)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def split_token(token: str) -> tuple[str, str] | None:
    """(key id, secret) from a token, or None if it is not shaped like one."""
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def secrets_match(secret: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_secret(secret), stored_hash)
