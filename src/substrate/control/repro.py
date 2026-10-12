"""What it takes to say "this result can be reproduced": a record of the environment, and a comparison that a rerun has to pass.

Two things are kept apart on purpose. The **software identity** (interpreter, library versions, a hash of this package's own source, the git commit
and image digest if there are any) decides whether a rerun happened in "the same" code; it is hashed into `environment_hash`. The **hardware and host**
(machine type, CPU count, platform string) are recorded but not hashed, because a result that depends on them is a finding in itself, not a different
experiment. A reproduction is judged on its numbers: every observable is compared with a tolerance, and the report says how far apart they are.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__

TRACKED_PACKAGES = ("numpy", "scipy", "PyYAML", "pyscf", "SQLAlchemy", "fastapi", "pydantic")
HASHED_KEYS = ("substrate", "python_version", "packages", "source_sha256", "image_digest")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def hash_tree(root: Path) -> str:
    """A hash of every Python file under `root` (relative path and bytes, in sorted order; bytecode caches excluded)."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def source_hash() -> str:
    """The hash of this package's own source, so two installs of "the same version" with different code are told apart."""
    return hash_tree(Path(__file__).resolve().parent.parent)


def git_commit(start: Path | None = None) -> str | None:
    """The commit the working tree is at, read from `.git` directly (no process is started); None outside a checkout."""
    here = (start or Path(__file__)).resolve()
    for directory in (here, *here.parents):
        git = directory / ".git"
        if git.is_dir():
            try:
                head = (git / "HEAD").read_text(encoding="utf-8").strip()
                if not head.startswith("ref:"):
                    return head if len(head) == 40 else None
                ref = head.split(None, 1)[1]
                loose = git / ref
                if loose.is_file():
                    return loose.read_text(encoding="utf-8").strip()
                packed = git / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(" " + ref):
                            return line.split()[0]
            except OSError:
                return None
            return None
    return None


def environment_fingerprint() -> dict:
    return {
        "substrate": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
        "source_sha256": source_hash(),
        "git_commit": git_commit(),
        "image_digest": os.environ.get("SUBSTRATE_IMAGE_DIGEST") or None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def environment_hash(fingerprint: dict) -> str:
    """Hash of the software identity only (see the module docstring); the interpreter counts to major.minor.patch."""
    subset = {key: fingerprint.get(key) for key in HASHED_KEYS}
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()


def environment_differences(a: dict, b: dict) -> dict[str, tuple]:
    """Every recorded key whose value differs, as {key: (a, b)}, including the ones that are not hashed."""
    keys = sorted(set(a) | set(b))
    return {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}


# -- results ----------------------------------------------------------------------------------------------------------------------
def observables_summary(observables: dict) -> dict:
    """The numeric content of a system's observables as plain JSON: {name: {"unit", "shape", "values"}} with values flattened."""
    out = {}
    for name in sorted(observables):
        quantity = observables[name]
        array = np.asarray(quantity.value, dtype=float)
        out[name] = {"unit": quantity.unit, "shape": list(array.shape), "values": [float(v) for v in array.ravel()]}
    return out


def _round_for_digest(values: list[float], digits: int = 10) -> list[str]:
    return [format(v, f".{digits - 1}e") if np.isfinite(v) else repr(v) for v in values]


def result_digest(summary: dict) -> str:
    """A hash of the summary with every number rounded to 10 significant digits: a quick way to see that two runs agree (almost) exactly, and nothing more. It cannot be
    robust against the noise that matters. Last-bit differences (multithreaded BLAS rounds differently from run to run) are amplified by the adaptive ODE solver to about
    1e-9, which changes a ten-digit digest, so two honest runs of the same experiment on the same machine can have different digests. The comparison below decides."""
    canonical = {name: {"unit": s["unit"], "shape": s["shape"], "values": _round_for_digest(s["values"])} for name, s in sorted(summary.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


@dataclass
class Comparison:
    matches: bool
    max_relative_difference: float
    differing: list[str] = field(default_factory=list)
    only_in_original: list[str] = field(default_factory=list)
    only_in_reproduction: list[str] = field(default_factory=list)
    worst: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"matches": self.matches, "max_relative_difference": self.max_relative_difference, "differing": self.differing,
                "only_in_original": self.only_in_original, "only_in_reproduction": self.only_in_reproduction, "worst": self.worst}


def compare_summaries(original: dict, reproduction: dict, *, rtol: float = 1e-8, floor: float = 1e-6) -> Comparison:
    """Compare two observable summaries name by name. Two values agree if |a - b| <= rtol * max(|a|, |b|, floor * M), where M is the largest magnitude in
    that observable. There is no absolute tolerance: observables span fifteen orders of magnitude (a relaxation time of 9e-11 s is as meaningful as a barrier
    of 0.5 eV), so any fixed number would swallow a real difference in one of them. The `floor` only keeps near-zero elements of an array (a wave function's
    tail) from being held to a relative standard against their own rounding noise. NaNs must be in the same places."""
    only_a = sorted(set(original) - set(reproduction))
    only_b = sorted(set(reproduction) - set(original))
    differing, worst = [], {}
    overall = 0.0
    for name in sorted(set(original) & set(reproduction)):
        a, b = original[name], reproduction[name]
        if a["shape"] != b["shape"] or a["unit"] != b["unit"]:
            differing.append(name)
            worst[name] = float("inf")
            overall = float("inf")
            continue
        x, y = np.asarray(a["values"], dtype=float), np.asarray(b["values"], dtype=float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            differing.append(name)
            worst[name] = float("inf")
            overall = float("inf")
            continue
        keep = ~np.isnan(x)
        x, y = x[keep], y[keep]
        if x.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            same = x == y                                        # equal infinities are equal; inf - inf would be NaN
            scale = np.maximum(np.abs(x), np.abs(y))
            difference = np.where(same, 0.0, np.abs(x - y))
            ratio = np.where(scale > 0, difference / np.where(scale > 0, scale, 1.0), 0.0)
            ratio = np.where(np.isnan(ratio), np.inf, ratio)     # a finite value against an infinite one is infinitely far apart
        relative = float(np.max(ratio))
        worst[name] = relative
        overall = max(overall, relative)
        allowed = rtol * np.maximum(scale, floor * (np.max(scale[np.isfinite(scale)]) if np.isfinite(scale).any() else 0.0))
        if not np.all(difference <= np.where(np.isfinite(scale), allowed, 0.0)):
            differing.append(name)
    return Comparison(not (differing or only_a or only_b), overall, differing, only_a, only_b, worst)
