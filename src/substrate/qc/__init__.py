"""Quantum-chemistry programs: the seam between substrate and codes that compute molecular energies.

An engine describes geometries as QCJobs; a QCProgram turns them into QCResults. Which program runs, and where, is
invisible to the engine. `PySCFProgram` runs PySCF in-process when it is installed, or through a worker in a Linux
environment reached with WSL (PySCF has no Windows build). Results are cached on disk by content, so an ensemble of a
hundred draws does not recompute a single energy twice.

Environment variables:
    SUBSTRATE_QC_MODE         "local" or "wsl" (default: local if pyscf imports, else wsl on Windows)
    SUBSTRATE_QC_WSL_DISTRO   distribution name for `wsl -d` (default: the default distribution)
    SUBSTRATE_QC_WSL_PYTHON   Python with PySCF inside WSL (default: $HOME/.substrate-qc/bin/python)
    SUBSTRATE_CACHE_DIR       where the energy cache lives (default: ~/.cache/substrate)

`scripts/setup_qc_env.sh` creates the WSL environment without sudo.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..errors import QCError, QCUnavailableError, ValidationError
from ..limits import require

#: bump when the meaning of a cached energy changes (new convergence settings, a bug fix, ...)
CACHE_VERSION = 1
CHUNK = 40                                  # jobs per worker process: bounds what a crash or timeout can lose
CC_CHUNK = 6                                # the same for coupled cluster, whose points cost minutes: a chunk must fit in the worker timeout


def chunk_size(jobs: Sequence["QCJob"]) -> int:
    """Jobs per worker process: small when any job is coupled cluster or a geometry relaxation (about a minute or more each), so a chunk
    finishes inside the timeout and a failure loses little."""
    return CC_CHUNK if any(_expensive(job) for job in jobs) else CHUNK


def _expensive(job: "QCJob") -> bool:
    """Coupled-cluster, relaxation and thermochemistry jobs cost minutes or more each; a single point costs seconds."""
    return job.theory.lower().startswith("ccsd") or job.task != "energy"


@dataclass(frozen=True)
class QCJob:
    """One calculation: geometry in angstrom, total charge, spin = 2S, and the level of theory. `task` is "energy" (a single point, the
    default), "relax" (minimise the energy over every atom's position, starting from this geometry, with analytic gradients; the result
    carries the relaxed geometry and the energy at the start) or "thermo" (relax tightly, then take the harmonic frequencies there; the
    result also carries the zero-point energy and the enthalpy at 298.15 K and 1 atm, rigid rotor and harmonic oscillator)."""

    atoms: tuple[tuple[str, float, float, float], ...]
    charge: int
    spin: int
    theory: str
    basis: str
    task: str = "energy"

    def key(self, program: str) -> str:
        """Content hash: same geometry (to 1e-8 angstrom), same method, same program -> same key."""
        canonical = {
            "v": CACHE_VERSION, "program": program, "theory": self.theory.lower(), "basis": self.basis.lower(),
            "charge": self.charge, "spin": self.spin,
            "atoms": [[s, round(x, 8), round(y, 8), round(z, 8)] for s, x, y, z in self.atoms],
        }
        if self.task != "energy":                    # added only when it differs, so every cached single point keeps its key
            canonical["task"] = self.task
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


#: What a theory or a basis-set name may contain. PySCF reads a basis string as a file path if it is not a known name, so a path separator, a dot
#: or a space has no business in one; none of the real names (6-31+G*, aug-cc-pV(T+d)Z, def2-SVP, b3lyp, wb97x-d) uses any of them.
_THEORY_NAME = re.compile(r"[A-Za-z0-9_:,+*.()\-]{1,80}")
_BASIS_NAME = re.compile(r"[A-Za-z0-9_,+*()\-]{1,40}")


def check_request(jobs: Sequence["QCJob"]) -> None:
    """Refuse a request that is too large before anything is looked up or computed (too many jobs, too many expensive ones, a job with too many
    atoms; the limits are in `substrate.limits` and come from the environment), or whose theory or basis is not a name."""
    for job in jobs:
        if not _THEORY_NAME.fullmatch(job.theory):
            raise ValidationError(f"theory {job.theory!r} is not a method name (letters, digits and : , + * . ( ) _ - only)")
        if not _BASIS_NAME.fullmatch(job.basis):
            raise ValidationError(f"basis {job.basis!r} is not a basis-set name (letters, digits and + * ( ) , _ - only: a file path is not accepted)")
    require("max_qc_jobs", len(jobs), "quantum-chemistry jobs in one request")
    require("max_expensive_qc_jobs", sum(1 for job in jobs if _expensive(job)), "coupled-cluster, relaxation and thermochemistry jobs in one request")
    require("max_atoms", max((len(job.atoms) for job in jobs), default=0), "atoms in one quantum-chemistry job")


@dataclass(frozen=True)
class QCResult:
    energy_hartree: float
    converged: bool
    homo_lumo_gap_hartree: float | None = None
    program: str = ""
    geometry: tuple[tuple[str, float, float, float], ...] | None = None      # a relaxation's final atoms, angstrom
    initial_energy_hartree: float | None = None                              # a relaxation's energy at the geometry it started from
    zero_point_hartree: float | None = None                                  # a thermo job's harmonic zero-point energy
    enthalpy_298_hartree: float | None = None                                # ... and its total enthalpy at 298.15 K (electronic energy included)

    def to_dict(self) -> dict:
        d = {"energy": self.energy_hartree, "converged": self.converged,
             "homo_lumo_gap": self.homo_lumo_gap_hartree, "program": self.program}
        if self.geometry is not None:
            d["geometry"] = [list(a) for a in self.geometry]
            d["initial_energy"] = self.initial_energy_hartree
        if self.enthalpy_298_hartree is not None:
            d["zero_point"] = self.zero_point_hartree
            d["enthalpy_298"] = self.enthalpy_298_hartree
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QCResult":
        geometry = None if d.get("geometry") is None else tuple((a[0], float(a[1]), float(a[2]), float(a[3])) for a in d["geometry"])
        return cls(d["energy"], d["converged"], d.get("homo_lumo_gap"), d.get("program", ""), geometry, d.get("initial_energy"),
                   d.get("zero_point"), d.get("enthalpy_298"))


class QCProgram(ABC):
    name: str

    @abstractmethod
    def compute(self, jobs: Sequence[QCJob]) -> list[QCResult]:
        """One result per job, in order. Raise QCError if the program itself fails; report SCF non-convergence through
        QCResult.converged instead."""


# -- cache ------------------------------------------------------------------------------------------------------------
def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class QCCache:
    """Energies on disk (SQLite), keyed by QCJob.key. Safe to share between runs and processes.

    Each entry is sealed: the stored text holds the result, a short hash of the worker source that produced it, and a SHA-256 digest of both. An
    entry whose digest does not match, whose text is not valid, or whose energy is not finite is never served (it counts in `rejected` and is
    recomputed and overwritten). That catches damage, hand edits and results left behind by a different version of the code; it does not stop
    someone who can write the file from sealing an entry again. Entries written before digests existed are served and reported as
    "unverified"; with `strict` (or SUBSTRATE_QC_CACHE_STRICT=1) only entries written by the current worker are served.
    """

    def __init__(self, path: str | Path | None = None, strict: bool | None = None):
        if path is None:
            root = Path(os.environ.get("SUBSTRATE_CACHE_DIR") or Path.home() / ".cache" / "substrate")
            path = root / "qc.sqlite3"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.strict = os.environ.get("SUBSTRATE_QC_CACHE_STRICT", "") not in ("", "0") if strict is None else strict
        self.rejected = 0                              # entries refused on reading, by this object
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS energies (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    @staticmethod
    def _seal(result: QCResult) -> str:
        body = {"result": result.to_dict(), "worker": worker_digest()}
        return json.dumps({**body, "sha256": _digest(body)})

    @staticmethod
    def _unseal(payload: str) -> tuple[QCResult | None, str, str | None]:
        """(result, state, worker): state is "verified", "unverified" (written before digests existed) or "corrupt" (then result is None)."""
        try:
            obj = json.loads(payload)
        except ValueError:
            return None, "corrupt", None
        if not isinstance(obj, dict):
            return None, "corrupt", None
        if "sha256" in obj:
            claimed = obj.pop("sha256")
            if claimed != _digest(obj):
                return None, "corrupt", None
            record, worker, state = obj.get("result"), obj.get("worker"), "verified"
        else:
            record, worker, state = obj, None, "unverified"
        try:
            result = QCResult.from_dict(record)
            finite = math.isfinite(result.energy_hartree)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return None, "corrupt", None
        return (result, state, worker) if finite else (None, "corrupt", None)

    def get_many(self, keys: Sequence[str]) -> dict[str, QCResult]:
        found: dict[str, QCResult] = {}
        with self._connect() as db:
            for start in range(0, len(keys), 500):
                batch = list(keys[start:start + 500])
                rows = db.execute(
                    f"SELECT key, payload FROM energies WHERE key IN ({','.join('?' * len(batch))})", batch).fetchall()
                for key, payload in rows:
                    result, state, worker = self._unseal(payload)
                    if result is None or (self.strict and (state != "verified" or worker != worker_digest())):
                        self.rejected += 1
                    else:
                        found[key] = result
        return found

    def put_many(self, items: dict[str, QCResult]) -> None:
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO energies (key, payload) VALUES (?, ?)",
                [(k, self._seal(r)) for k, r in items.items()],
            )

    def audit(self) -> dict:
        """Every entry checked: counts by state, and (for the readable ones) by the program version and the worker that wrote them."""
        states, programs, workers = Counter(), Counter(), Counter()
        with self._connect() as db:
            for (payload,) in db.execute("SELECT payload FROM energies"):
                result, state, worker = self._unseal(payload)
                states[state] += 1
                if result is not None:
                    programs[result.program or "(unknown)"] += 1
                    if worker:
                        workers[worker] += 1
        return {"total": sum(states.values()), "verified": states["verified"], "unverified": states["unverified"], "corrupt": states["corrupt"],
                "by_program": dict(programs), "by_worker": dict(workers)}

    def purge(self, *, corrupt: bool = False, unverified: bool = False, worker: str | None = None) -> int:
        """Delete entries that fail their digest (`corrupt`), that predate digests (`unverified`), or that a given worker digest wrote. Returns how many."""
        doomed = []
        with self._connect() as db:
            for key, payload in db.execute("SELECT key, payload FROM energies").fetchall():
                result, state, entry_worker = self._unseal(payload)
                if (corrupt and state == "corrupt") or (unverified and state == "unverified") or (worker is not None and entry_worker == worker):
                    doomed.append((key,))
            db.executemany("DELETE FROM energies WHERE key = ?", doomed)
        return len(doomed)

    def __len__(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM energies").fetchone()[0]


def compute_cached(program: QCProgram, jobs: Sequence[QCJob], cache: QCCache | None = None) -> tuple[list[QCResult], int, int]:
    """Results for every job, computing only what the cache lacks. Returns (results, n_computed, n_cached).
    Duplicate jobs in the request are computed once."""
    check_request(jobs)                                    # before the cache is even opened
    cache = cache if cache is not None else QCCache()      # NOT `cache or ...`: an empty cache is falsy (it has __len__)
    keys = [job.key(program.name) for job in jobs]
    have = cache.get_many(list(set(keys)))
    missing: dict[str, QCJob] = {k: j for k, j in zip(keys, jobs) if k not in have}
    n_cached = len(set(keys)) - len(missing)
    size = chunk_size(list(missing.values()))
    for start in range(0, len(missing), size):
        chunk = list(missing.items())[start:start + size]
        results = program.compute([job for _, job in chunk])
        if len(results) != len(chunk):
            raise QCError(f"{program.name} returned {len(results)} results for {len(chunk)} jobs")
        fresh = {key: result for (key, _), result in zip(chunk, results) if result.converged}
        cache.put_many(fresh)                      # only converged energies are worth keeping; save progress per chunk
        have.update({key: result for (key, _), result in zip(chunk, results)})
    return [have[k] for k in keys], len(missing), n_cached


# -- PySCF ------------------------------------------------------------------------------------------------------------
_WORKER = Path(__file__).with_name("pyscf_worker.py")


@functools.lru_cache(maxsize=1)
def worker_digest() -> str:
    """A short hash of the worker's source, stored with every cached result so that results from a different version of the worker can be found."""
    return hashlib.sha256(_WORKER.read_bytes()).hexdigest()[:16]


def _to_wsl_path(path: Path) -> str:
    """C:\\Users\\me\\x  ->  /mnt/c/Users/me/x"""
    path = path.resolve()
    drive, rest = path.drive, path.as_posix()[len(path.drive):]
    if len(drive) != 2 or drive[1] != ":":
        raise QCError(f"cannot map {path} into WSL (only drive-letter paths are supported)")
    return f"/mnt/{drive[0].lower()}{rest}"


class PySCFProgram(QCProgram):
    name = "pyscf"

    def __init__(self, mode: str | None = None, timeout: float = 3600.0):
        self._mode = mode or os.environ.get("SUBSTRATE_QC_MODE")
        self.timeout = timeout
        self.distro = os.environ.get("SUBSTRATE_QC_WSL_DISTRO")
        self.wsl_python = os.environ.get("SUBSTRATE_QC_WSL_PYTHON")      # None -> $HOME/.substrate-qc/bin/python
        self.version: str | None = None

    # -- discovery ----------------------------------------------------------------------------------------------------
    def _wsl_command(self, script: str) -> list[str]:
        return ["wsl.exe", *(["-d", self.distro] if self.distro else []), "--", "bash", "-c", script]

    def _python_in_wsl(self) -> str:
        return shlex.quote(self.wsl_python) if self.wsl_python else '"$HOME/.substrate-qc/bin/python"'

    def _wsl_has_pyscf(self) -> bool:
        if not shutil.which("wsl.exe"):
            return False
        probe = f'{self._python_in_wsl()} -c "import pyscf"'
        try:
            return subprocess.run(self._wsl_command(probe), capture_output=True, timeout=120).returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def mode(self) -> str:
        if self._mode is None:
            if importlib.util.find_spec("pyscf") is not None:
                self._mode = "local"
            elif sys.platform == "win32" and self._wsl_has_pyscf():
                self._mode = "wsl"
            else:
                raise QCUnavailableError(
                    "PySCF is not available. Install it (pip install pyscf) on Linux/macOS, or on Windows create the "
                    "WSL environment with scripts/setup_qc_env.sh (see the README)."
                )
        if self._mode not in ("local", "wsl"):
            raise QCError(f"unknown SUBSTRATE_QC_MODE '{self._mode}' (use 'local' or 'wsl')")
        return self._mode

    def available(self) -> bool:
        try:
            self.mode()
            return True
        except QCUnavailableError:
            return False

    # -- execution ----------------------------------------------------------------------------------------------------
    def compute(self, jobs: Sequence[QCJob]) -> list[QCResult]:
        requests = [
            {"id": i, "atoms": [list(a) for a in job.atoms], "charge": job.charge, "spin": job.spin,
             "theory": job.theory, "basis": job.basis, "task": job.task}
            for i, job in enumerate(jobs)
        ]
        raw = self._run_local(requests) if self.mode() == "local" else self._run_wsl(requests)
        by_id = {r["id"]: r for r in raw if "id" in r}
        results = []
        for i in range(len(jobs)):
            r = by_id.get(i)
            if r is None:
                raise QCError(f"PySCF worker returned no result for job {i}")
            if "error" in r:
                raise QCError(f"PySCF failed on job {i}: {r['error']}")
            results.append(QCResult.from_dict({**r, "program": self.version or "pyscf"}))
        return results

    def _run_local(self, requests: list[dict]) -> list[dict]:
        from .pyscf_worker import run_job
        import pyscf
        self.version = f"pyscf {pyscf.__version__}"
        out = []
        for request in requests:
            try:
                out.append(run_job(request))
            except Exception as error:
                out.append({"id": request["id"], "error": f"{type(error).__name__}: {error}"})
        return out

    def _run_wsl(self, requests: list[dict]) -> list[dict]:
        command = self._wsl_command(f"exec {self._python_in_wsl()} {shlex.quote(_to_wsl_path(_WORKER))}")
        try:
            proc = subprocess.run(command, input=json.dumps({"jobs": requests}).encode(), capture_output=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise QCError(f"PySCF worker exceeded the {self.timeout:g} s timeout on {len(requests)} jobs") from None
        results, done = [], None
        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue                                       # banners and warnings are not protocol
            if record.get("done"):
                done = record
            else:
                results.append(record)
        if proc.returncode != 0 or done is None:
            tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-5:]
            raise QCError(f"PySCF worker failed (exit {proc.returncode}): " + " | ".join(tail))
        self.version = done.get("program")
        return results


# -- registry ---------------------------------------------------------------------------------------------------------
_PROGRAMS: dict[str, Callable[[], QCProgram]] = {"pyscf": PySCFProgram}


def register_program(name: str, factory: Callable[[], QCProgram]) -> None:
    _PROGRAMS[name] = factory


def get_program(name: str = "pyscf") -> QCProgram:
    try:
        return _PROGRAMS[name]()
    except KeyError:
        raise QCError(f"no quantum-chemistry program '{name}' (registered: {', '.join(_PROGRAMS)})") from None
