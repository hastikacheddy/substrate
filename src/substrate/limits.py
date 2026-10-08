"""Bounds on what one run may ask for, and the checks that keep a number from becoming an enormous request.

Scientific code is easy to point at something huge: a scan of 10^6 x 10^6 points, a million-draw ensemble, a thousand-atom molecule sent to a
quantum-chemistry program, a step count of NaN. These limits refuse such a request before anything is allocated or computed, with a message that
names the limit and how to change it.

The limits come from the environment (`SUBSTRATE_MAX_...`), never from an experiment file: the file is the thing being checked, so it cannot be
allowed to raise its own ceiling. The defaults are several times larger than anything the studies in this repository need (the largest single
scan is 324 energies, the largest ensemble 2,000 draws, the largest molecule 12 atoms).
"""
from __future__ import annotations

import math
import os

from .errors import ResourceLimitError, ValidationError

DEFAULTS: dict[str, int] = {
    "max_ensemble": 2_000,                 # Monte Carlo draws in one run
    "max_grid_points": 1_000_000,          # points of one scan, grid or time axis (the product, for a two-dimensional grid)
    "max_qc_jobs": 2_000,                  # quantum-chemistry jobs in one request (cached or not)
    "max_expensive_qc_jobs": 200,          # of those, coupled-cluster, relaxation and thermochemistry jobs, which cost minutes each
    "max_atoms": 50,                       # atoms in one quantum-chemistry job
    "max_references": 60,                  # surfaces in one transfer study (its cost grows with the square)
    "max_spec_bytes": 1_000_000,           # size of an experiment file
}


def _variable(name: str) -> str:
    return "SUBSTRATE_" + name.upper()


def limit(name: str) -> int:
    """The current value of a limit: the environment variable if it is set, else the default."""
    if name not in DEFAULTS:
        raise KeyError(f"unknown limit '{name}' (have: {', '.join(DEFAULTS)})")
    raw = os.environ.get(_variable(name), "").strip()
    if not raw:
        return DEFAULTS[name]
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ResourceLimitError(f"{_variable(name)} must be a positive whole number (got '{raw}')")
    return value


def require(name: str, requested: int, what: str) -> None:
    """Refuse `requested` (a count) when it exceeds the limit `name`; `what` says what is being counted, for the message."""
    maximum = limit(name)
    if requested > maximum:
        raise ResourceLimitError(f"{what}: {requested:,} exceeds the limit of {maximum:,} (set {_variable(name)} to raise it)")


def whole_number(owner: str, name: str, value, *, minimum: int | None = None, maximum: str | None = None) -> int:
    """`value` as an int, for a count such as a number of grid points or samples. Refuses a bool, NaN, infinity and a fraction as invalid input
    (a ValidationError), as it does a value below `minimum`, and, with `maximum` naming a limit, a value above it (a ResourceLimitError)."""
    if isinstance(value, bool) or not (isinstance(value, (int, float)) or hasattr(value, "__float__")):
        raise ValidationError(f"{owner}: {name} must be a whole number (got {value!r})")
    number = float(value)
    if not math.isfinite(number) or number != math.floor(number):
        raise ValidationError(f"{owner}: {name} must be a finite whole number (got {value!r})")
    count = int(number)
    if minimum is not None and count < minimum:
        raise ValidationError(f"{owner}: {name} must be at least {minimum} (got {count})")
    if maximum is not None:
        require(maximum, count, f"{owner}: {name}")
    return count


def finite(owner: str, name: str, value) -> None:
    """Refuse NaN and infinity (in a scalar or an array) as invalid input."""
    import numpy as np
    try:
        ok = bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
    except (TypeError, ValueError):
        raise ValidationError(f"{owner}: {name} is not a number (got {value!r})") from None
    if not ok:
        raise ValidationError(f"{owner}: {name} must be finite (got {value!r})")
