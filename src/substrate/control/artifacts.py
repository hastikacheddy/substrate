"""Where results live: an immutable, content-addressed store with its metadata in the registry.

Large things (a trajectory, a surface, a stored system, a report) do not go in the database. They go in an artifact store under the SHA-256 of their
bytes, inside the owning tenant's namespace, and the registry keeps a pointer with the size, type and name. That gives three properties for free:
an artifact cannot be changed after it is written (the same bytes have the same name; different bytes are a different artifact), every read can be
checked against the name it was asked for, and writing the same result twice stores it once.

What is accepted is deliberately narrow: JSON, plain text, CSV, Markdown and NumPy arrays (loaded with `allow_pickle=False`). Nothing here ever
deserialises a pickle, and an artifact whose bytes do not match its stated type is refused. An optional scanner hook lets a deployment add its own
checks (antivirus, content policy) before a write.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ..errors import SubstrateError, ValidationError
from ..limits import require
from .registry import Registry
from .security import Action, Principal, QuotaExceededError

MEDIA_TYPES = {"application/json", "text/plain", "text/markdown", "text/csv", "application/x-npy"}
_SHA = re.compile(r"[0-9a-f]{64}")
_TENANT = re.compile(r"[A-Za-z0-9_\-]{1,32}")


class ArtifactError(SubstrateError):
    """An artifact is missing, is not what it claims to be, or has been altered."""


@dataclass(frozen=True)
class ArtifactRef:
    tenant_id: str
    sha256: str
    size: int
    media_type: str
    name: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_content(data: bytes, media_type: str) -> None:
    """Refuse bytes that are not what `media_type` says they are, or that are of a type the store does not accept."""
    if media_type not in MEDIA_TYPES:
        raise ValidationError(f"artifact type '{media_type}' is not accepted (have: {', '.join(sorted(MEDIA_TYPES))})")
    if media_type == "application/json":
        try:
            json.loads(data.decode("utf-8"), parse_constant=_no_constants)
        except (ValueError, RecursionError) as error:
            raise ValidationError(f"the artifact is not valid JSON ({error})") from None
    elif media_type == "application/x-npy":
        try:
            array = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as error:                              # noqa: BLE001 (numpy raises ValueError, OSError, EOFError, ... for a bad file)
            raise ValidationError(f"the artifact is not a NumPy array without pickled objects ({type(error).__name__}: {error})") from None
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValidationError("the artifact is not a plain NumPy array")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValidationError("the artifact is not UTF-8 text") from None
        if "\x00" in text:
            raise ValidationError("the artifact contains NUL bytes")


def encode_nonfinite(value):
    """Replace every NaN and infinity in nested lists and dicts by a tagged string, so the result is strict JSON."""
    if isinstance(value, float):
        if value != value:
            return {"__float__": "nan"}
        if value in (float("inf"), float("-inf")):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, dict):
        return {k: encode_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_nonfinite(v) for v in value]
    return value


def decode_nonfinite(value):
    if isinstance(value, dict):
        if set(value) == {"__float__"} and value["__float__"] in ("nan", "inf", "-inf"):
            return float(value["__float__"])
        return {k: decode_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_nonfinite(v) for v in value]
    return value


def _no_constants(name: str):
    raise ValueError(f"'{name}' is not allowed in stored JSON")


class ArtifactStore(Protocol):
    def put(self, tenant_id: str, data: bytes) -> str: ...
    def get(self, tenant_id: str, sha256: str) -> bytes: ...
    def exists(self, tenant_id: str, sha256: str) -> bool: ...


def _read(path: Path) -> bytes:
    """The file's bytes. On Windows a file that another writer is renaming over at that moment cannot be opened for a few milliseconds, so a
    PermissionError is retried briefly before it is believed (a missing file is not retried)."""
    for attempt in range(12):
        try:
            return path.read_bytes()
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise AssertionError("unreachable")


class LocalArtifactStore:
    """Files under `root/<tenant>/<aa>/<bb>/<sha256>`. Written to a temporary name and renamed, so a reader never sees half a file."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, tenant_id: str, sha256: str) -> Path:
        if not _TENANT.fullmatch(tenant_id) or not _SHA.fullmatch(sha256):
            raise ArtifactError("malformed artifact address")           # also what stops a path from escaping the store
        return self.root / tenant_id / sha256[:2] / sha256[2:4] / sha256

    def put(self, tenant_id: str, data: bytes) -> str:
        sha = sha256_bytes(data)
        path = self._path(tenant_id, sha)
        if path.exists():
            if sha256_bytes(_read(path)) != sha:
                raise ArtifactError(f"the stored copy of {sha} does not match its name: the store has been altered")
            return sha
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{sha}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")      # unique per writer: two threads must not share a temporary file
        temporary.write_bytes(data)
        try:
            for attempt in range(12):
                try:
                    os.replace(temporary, path)
                    break
                except OSError:
                    # On Windows a rename onto a file another writer has open fails with a sharing violation. The name is the hash of the content, so if the
                    # destination is there and holds exactly these bytes, this write has already happened (someone else did it) and there is nothing to retry.
                    try:
                        if path.is_file() and sha256_bytes(_read(path)) == sha:
                            break
                    except OSError:
                        pass
                    if attempt == 11:
                        raise
                    time.sleep(0.005 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
        return sha

    def get(self, tenant_id: str, sha256: str) -> bytes:
        path = self._path(tenant_id, sha256)
        try:
            data = _read(path)
        except FileNotFoundError:
            raise ArtifactError(f"no artifact {sha256}") from None
        if sha256_bytes(data) != sha256:
            raise ArtifactError(f"artifact {sha256} does not match its checksum: the stored bytes have changed")
        return data

    def exists(self, tenant_id: str, sha256: str) -> bool:
        return self._path(tenant_id, sha256).is_file()


class S3ArtifactStore:
    """The same contract on an S3-compatible service (MinIO, AWS). Takes any client with `put_object`, `get_object` and `head_object`, such as boto3's.

    Objects are keyed by tenant and checksum and are never overwritten with different bytes; a read is checked against the checksum in its key.
    """

    def __init__(self, client, bucket: str, prefix: str = "artifacts"):
        self.client, self.bucket, self.prefix = client, bucket, prefix.strip("/")

    @classmethod
    def from_boto3(cls, bucket: str, prefix: str = "artifacts", **client_options):
        import boto3                                              # optional dependency, imported only when this store is asked for
        return cls(boto3.client("s3", **client_options), bucket, prefix)

    def _key(self, tenant_id: str, sha256: str) -> str:
        if not _TENANT.fullmatch(tenant_id) or not _SHA.fullmatch(sha256):
            raise ArtifactError("malformed artifact address")
        return f"{self.prefix}/{tenant_id}/{sha256[:2]}/{sha256}"

    def put(self, tenant_id: str, data: bytes) -> str:
        sha = sha256_bytes(data)
        if not self.exists(tenant_id, sha):
            self.client.put_object(Bucket=self.bucket, Key=self._key(tenant_id, sha), Body=data)
        return sha

    def get(self, tenant_id: str, sha256: str) -> bytes:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key(tenant_id, sha256))["Body"].read()
        except Exception as error:                                # noqa: BLE001 (boto3 raises its own ClientError)
            if type(error).__name__ in ("NoSuchKey", "ClientError", "KeyError"):
                raise ArtifactError(f"no artifact {sha256}") from None
            raise
        if sha256_bytes(body) != sha256:
            raise ArtifactError(f"artifact {sha256} does not match its checksum: the stored bytes have changed")
        return body

    def exists(self, tenant_id: str, sha256: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(tenant_id, sha256))
            return True
        except Exception as error:                                # noqa: BLE001
            if type(error).__name__ in ("NoSuchKey", "ClientError", "KeyError"):
                return False
            raise


class Artifacts:
    """The store plus the registry's record of it, with the checks that belong at the boundary: type, size, quota, tenant and role."""

    def __init__(self, registry: Registry, store: ArtifactStore, scanner: Callable[[bytes, str], None] | None = None):
        self.registry, self.store, self.scanner = registry, store, scanner

    def write(self, tenant_id: str, data: bytes, *, name: str, media_type: str, run_id: str | None = None) -> ArtifactRef:
        require("max_artifact_bytes", len(data), f"artifact '{name}': size in bytes")
        check_content(data, media_type)
        if self.scanner is not None:
            self.scanner(data, media_type)                         # raises to refuse
        used, allowed = self.registry.artifact_bytes_total(tenant_id), self.registry.quota(tenant_id, "artifact_bytes_total")
        sha = sha256_bytes(data)
        if self.registry.artifact_meta_in_tenant(tenant_id, sha) is None and used + len(data) > allowed:
            self.registry.audit(tenant_id, "quota.blocked", metric="artifact_bytes_total", requested=len(data), used=used, limit=allowed, result="BLOCKED")
            raise QuotaExceededError(f"storing '{name}' would take the tenant to {used + len(data):,} of {allowed:,.0f} bytes")
        self.store.put(tenant_id, data)
        self.registry.record_artifact(tenant_id, sha, run_id=run_id, name=name, media_type=media_type, size=len(data))
        return ArtifactRef(tenant_id, sha, len(data), media_type, name)

    def write_json(self, tenant_id: str, value, *, name: str, run_id: str | None = None) -> ArtifactRef:
        """Store JSON data. NaN and infinity, which JSON cannot express, are written as {"__float__": "nan"} and come back as floats on `read_json`."""
        data = json.dumps(encode_nonfinite(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return self.write(tenant_id, data, name=name, media_type="application/json", run_id=run_id)

    def write_array(self, tenant_id: str, array: np.ndarray, *, name: str, run_id: str | None = None) -> ArtifactRef:
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return self.write(tenant_id, buffer.getvalue(), name=name, media_type="application/x-npy", run_id=run_id)

    def read_for(self, tenant_id: str, sha256: str) -> bytes:
        """For the orchestrator and workers, acting for a tenant: the bytes, only if the registry knows this tenant owns the artifact."""
        if self.registry.artifact_meta_in_tenant(tenant_id, sha256) is None:
            raise ArtifactError(f"no artifact {sha256}")
        return self.store.get(tenant_id, sha256)

    def read(self, principal: Principal, sha256: str) -> tuple[bytes, dict]:
        principal.require(Action.ARTIFACT_READ)
        meta = self.registry.artifact_meta(principal, sha256)
        if meta is None:
            raise ArtifactError(f"no artifact {sha256}")
        return self.store.get(principal.tenant_id, sha256), meta

    def read_json(self, tenant_id: str, sha256: str):
        return decode_nonfinite(json.loads(self.read_for(tenant_id, sha256).decode("utf-8")))

    def read_array(self, tenant_id: str, sha256: str) -> np.ndarray:
        return np.load(io.BytesIO(self.read_for(tenant_id, sha256)), allow_pickle=False)
