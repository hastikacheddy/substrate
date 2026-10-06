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

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..errors import QCError, QCUnavailableError

#: bump when the meaning of a cached energy changes (new convergence settings, a bug fix, ...)
CACHE_VERSION = 1
CHUNK = 40                                  # jobs per worker process: bounds what a crash or timeout can lose


@dataclass(frozen=True)
class QCJob:
    """One single-point energy: geometry in angstrom, total charge, spin = 2S, and the level of theory."""

    atoms: tuple[tuple[str, float, float, float], ...]
    charge: int
    spin: int
    theory: str
    basis: str

    def key(self, program: str) -> str:
        """Content hash: same geometry (to 1e-8 angstrom), same method, same program -> same key."""
        canonical = {
            "v": CACHE_VERSION, "program": program, "theory": self.theory.lower(), "basis": self.basis.lower(),
            "charge": self.charge, "spin": self.spin,
            "atoms": [[s, round(x, 8), round(y, 8), round(z, 8)] for s, x, y, z in self.atoms],
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class QCResult:
    energy_hartree: float
    converged: bool
    homo_lumo_gap_hartree: float | None = None
    program: str = ""

    def to_dict(self) -> dict:
        return {"energy": self.energy_hartree, "converged": self.converged,
                "homo_lumo_gap": self.homo_lumo_gap_hartree, "program": self.program}

    @classmethod
    def from_dict(cls, d: dict) -> "QCResult":
        return cls(d["energy"], d["converged"], d.get("homo_lumo_gap"), d.get("program", ""))


class QCProgram(ABC):
    name: str

    @abstractmethod
    def compute(self, jobs: Sequence[QCJob]) -> list[QCResult]:
        """One result per job, in order. Raise QCError if the program itself fails; report SCF non-convergence through
        QCResult.converged instead."""


# -- cache ------------------------------------------------------------------------------------------------------------
class QCCache:
    """Energies on disk (SQLite), keyed by QCJob.key. Safe to share between runs and processes."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            root = Path(os.environ.get("SUBSTRATE_CACHE_DIR") or Path.home() / ".cache" / "substrate")
            path = root / "qc.sqlite3"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS energies (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get_many(self, keys: Sequence[str]) -> dict[str, QCResult]:
        found: dict[str, QCResult] = {}
        with self._connect() as db:
            for start in range(0, len(keys), 500):
                batch = list(keys[start:start + 500])
                rows = db.execute(
                    f"SELECT key, payload FROM energies WHERE key IN ({','.join('?' * len(batch))})", batch).fetchall()
                found.update({k: QCResult.from_dict(json.loads(p)) for k, p in rows})
        return found

    def put_many(self, items: dict[str, QCResult]) -> None:
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO energies (key, payload) VALUES (?, ?)",
                [(k, json.dumps(r.to_dict())) for k, r in items.items()],
            )

    def __len__(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM energies").fetchone()[0]


def compute_cached(program: QCProgram, jobs: Sequence[QCJob], cache: QCCache | None = None) -> tuple[list[QCResult], int, int]:
    """Results for every job, computing only what the cache lacks. Returns (results, n_computed, n_cached).
    Duplicate jobs in the request are computed once."""
    cache = cache if cache is not None else QCCache()      # NOT `cache or ...`: an empty cache is falsy (it has __len__)
    keys = [job.key(program.name) for job in jobs]
    have = cache.get_many(list(set(keys)))
    missing: dict[str, QCJob] = {k: j for k, j in zip(keys, jobs) if k not in have}
    n_cached = len(set(keys)) - len(missing)
    for start in range(0, len(missing), CHUNK):
        chunk = list(missing.items())[start:start + CHUNK]
        results = program.compute([job for _, job in chunk])
        if len(results) != len(chunk):
            raise QCError(f"{program.name} returned {len(results)} results for {len(chunk)} jobs")
        fresh = {key: result for (key, _), result in zip(chunk, results) if result.converged}
        cache.put_many(fresh)                      # only converged energies are worth keeping; save progress per chunk
        have.update({key: result for (key, _), result in zip(chunk, results)})
    return [have[k] for k in keys], len(missing), n_cached


# -- PySCF ------------------------------------------------------------------------------------------------------------
_WORKER = Path(__file__).with_name("pyscf_worker.py")


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
             "theory": job.theory, "basis": job.basis}
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
            results.append(QCResult(r["energy"], r["converged"], r.get("homo_lumo_gap"), self.version or "pyscf"))
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
