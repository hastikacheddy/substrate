"""Running a step in a separate process, with a deadline, resource caps and nothing it should not see.

The model code that runs a step is the part of the platform that does the most arithmetic on the least-trusted input (an experiment someone submitted). The
worker keeps the database, the artifact store and every credential to itself and hands the step plain JSON; the step runs in a fresh interpreter that

  * is killed if it has not answered by the deadline,
  * on POSIX, cannot use more than a set amount of memory or CPU time (RLIMIT_AS, RLIMIT_CPU) and cannot write large files (RLIMIT_FSIZE),
  * starts with every environment variable that looks like a secret removed (database URLs, keys, tokens, cloud credentials),
  * answers over a pipe with **JSON bytes only**: the parent never unpickles anything the child sends, so a compromised child cannot run code in the worker.

Windows has no equivalent of the rlimits here, so on Windows only the deadline and the environment scrub apply; a deployment gets the rest from its container
(memory and CPU limits, a read-only root filesystem, no network except an allowlist). This is process isolation, not a security boundary against a determined
attacker with arbitrary code execution: that needs a container or a micro-VM around the worker as well.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import re

from dataclasses import asdict

from .. import errors as substrate_errors
from ..errors import SubstrateError
from ..pipeline import Pipeline
from ..qpu import QuantumPolicy
from .steps import StepRequest, apply_step_data

SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION)", re.IGNORECASE)
SECRET_PREFIXES = ("DATABASE_", "AWS_", "AZURE_", "GOOGLE_", "GCP_", "IBM_", "QISKIT_", "REDIS_", "MINIO_", "SUBSTRATE_DB", "SUBSTRATE_S3", "PG")
KEEP = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}


class StepTimeout(TimeoutError):
    """The step did not answer by its deadline and was killed. A timeout is worth another try (the machine may have been busy)."""


class StepCrashed(RuntimeError):
    """The step's process ended without answering (killed, out of memory, a hard crash)."""


def scrub_environment(env: dict[str, str]) -> dict[str, str]:
    """`env` without anything that looks like a credential. Names on the keep-list survive even if they match a pattern."""
    clean = {}
    for name, value in env.items():
        upper = name.upper()
        if upper in KEEP or upper.startswith("SUBSTRATE_MAX_") or upper == "SUBSTRATE_IMAGE_DIGEST":
            clean[name] = value
        elif SECRET_NAME.search(name) or upper.startswith(SECRET_PREFIXES):
            continue
        else:
            clean[name] = value
    return clean


def _apply_limits(memory_bytes: int | None, cpu_seconds: int | None) -> None:
    try:
        import resource
    except ImportError:                                       # Windows: no rlimits
        return
    for which, value in ((resource.RLIMIT_AS, memory_bytes), (resource.RLIMIT_CPU, cpu_seconds), (resource.RLIMIT_FSIZE, 10 * 1024 * 1024)):
        if value:
            _, hard = resource.getrlimit(which)
            resource.setrlimit(which, (value if hard in (-1, resource.RLIM_INFINITY) else min(value, hard), hard))


def _isolate_environment() -> None:
    """Leave this process with the scrubbed environment and nothing else: what it was started with, credentials included, is gone."""
    scrubbed = scrub_environment(dict(os.environ))
    os.environ.clear()
    os.environ.update(scrubbed)


def _child(connection, payload: str) -> None:
    """The step's process. Anything it says goes back as one JSON message."""
    try:
        settings = json.loads(payload)
        _isolate_environment()
        _apply_limits(settings["memory_bytes"], settings["cpu_seconds"])
        from ..defaults import default_registry
        policy = QuantumPolicy(**{k: tuple(v) if isinstance(v, list) else v for k, v in settings["policy"].items()}) if settings.get("policy") else None
        out = apply_step_data(Pipeline(default_registry()), StepRequest(**settings["request"]), policy).to_dict()
        message = {"status": "ok", "system": out}
    except MemoryError:
        message = {"status": "error", "type": "ResourceLimitError", "message": "the step exceeded its memory limit"}
    except BaseException as error:                            # noqa: BLE001 (whatever happened is reported, never raised into a pipe nobody reads)
        message = {"status": "error", "type": type(error).__name__, "message": str(error)[:2000]}
    try:
        connection.send_bytes(json.dumps(message).encode())
    finally:
        connection.close()


def _rehydrate(kind: str, message: str) -> BaseException:
    """The exception a child reported, as the same class if it is one of ours (so failure classification still works), else a RuntimeError."""
    from .. import qpu
    from . import security
    cls = next((c for c in (getattr(m, kind, None) for m in (substrate_errors, qpu, security)) if isinstance(c, type) and issubclass(c, SubstrateError)), None)
    if cls is not None:
        try:
            return cls(message)
        except TypeError:
            return SubstrateError(message)
    return RuntimeError(f"{kind}: {message}")


class ProcessExecutor:
    def __init__(self, *, memory_bytes: int | None = 4 * 1024**3, cpu_seconds: int | None = None, target=_child, start_method: str = "spawn",
                 quantum_policy: QuantumPolicy | None = None):
        self.memory_bytes, self.cpu_seconds, self.target = memory_bytes, cpu_seconds, target
        self.quantum_policy = quantum_policy
        self.context = multiprocessing.get_context(start_method)
        self.last_process = None

    def apply(self, pipeline: Pipeline, request: StepRequest) -> dict:
        """The step's output system as data. `pipeline` is ignored on purpose: the child builds its own, so nothing of the worker's state is shared."""
        receiver, sender = self.context.Pipe(duplex=False)
        payload = json.dumps({"request": request.to_dict(), "memory_bytes": self.memory_bytes, "cpu_seconds": self.cpu_seconds,
                              "policy": asdict(self.quantum_policy) if self.quantum_policy else None})
        process = self.context.Process(target=self.target, args=(sender, payload), daemon=True)
        process.start()
        self.last_process = process
        sender.close()
        try:
            if not receiver.poll(request.timeout_s):
                process.kill()
                raise StepTimeout(f"the step did not finish within {request.timeout_s:g} s and was stopped")
            try:
                raw = receiver.recv_bytes()                    # bytes, never recv(): nothing the child sends is ever unpickled
            except EOFError:
                process.join(5)
                raise StepCrashed(f"the step's process ended without a result (exit code {process.exitcode})") from None
        finally:
            receiver.close()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise StepCrashed("the step's process sent something that is not a result") from None
        if not isinstance(message, dict) or message.get("status") not in ("ok", "error"):
            raise StepCrashed("the step's process sent something that is not a result")
        if message["status"] == "error":
            raise _rehydrate(str(message.get("type", "")), str(message.get("message", "")))
        return message["system"]


def default_isolation() -> str:
    return os.environ.get("SUBSTRATE_WORKER_ISOLATION", "inline")


__all__ = ["ProcessExecutor", "StepTimeout", "StepCrashed", "scrub_environment", "default_isolation"]
