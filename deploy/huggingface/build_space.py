"""Assemble a Hugging Face Space (Docker) that runs Mission control, the local GUI, as a public demo.

    python deploy/huggingface/build_space.py OUTDIR --space-id Hastika06/substrate-mission-control --cache-dir ~/.cache/substrate

What goes in the folder, and nothing else:

* the package source, the experiments folder and the locked runtime versions, as they are in git at HEAD (so the Space is a snapshot of a commit, which the card names);
* the energy cache from `--cache-dir`, copied with SQLite's backup API (a consistent copy, no write-ahead files) and **scanned first**: the build stops if any stored value
  contains a user name, a drive or home path or a host name, so nothing from this machine is published by accident;
* a Dockerfile (a non-root user, port 7860, the Space's own host name as the only extra host the GUI answers to, two jobs at once, a smaller ensemble limit) that also
  computes the full transfer study from the stored energies while the image is built (prewarm.py), and the Space's README card (license "other": none has been chosen).
  Saved studies are not copied from this machine: they are keyed to the exact bytes of the reference files and go stale the moment one differs.

It refuses to write into a folder that is not empty. It never reads a token and does not talk to Hugging Face; `upload_space.py` does that.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}")
TRACKED = ("src", "experiments", "pyproject.toml", "requirements-lock.txt")
GITHUB = "https://github.com/hastikacheddy/substrate"
PORT = 7860
MAX_JOBS = 2
MAX_ENSEMBLE = 200                       # draws in one run (the default is 2,000): a visitor shares a small CPU with everyone else


class BuildError(Exception):
    """Something that makes the Space unsafe or impossible to build. The message says what and how to fix it."""


def space_host(space_id: str) -> str:
    """The host name a Space is served from: owner and name joined by a hyphen, lower-case, dots and underscores turned into hyphens."""
    if not SPACE_ID.fullmatch(space_id):
        raise BuildError(f"'{space_id}' is not a Space id: use owner/name (letters, digits, - _ .)")
    owner, name = space_id.split("/")
    return f"{owner}-{name}".lower().replace("_", "-").replace(".", "-") + ".hf.space"


def machine_markers(extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Strings that would show this machine or its owner in published data: the account name, home folder, drive and home paths, local host names."""
    home = Path.home()
    base = (home.name.lower(), str(home).lower().replace("\\", "/"), "c:\\", "c:/", "/home/", "\\users\\", "/users/", "localhost", "127.0.0.1")
    return tuple(m for m in base + tuple(e.lower() for e in extra) if m)


def _tracked(paths=TRACKED) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *paths], cwd=ROOT, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise BuildError("this needs a git checkout: it ships the files git tracks, not whatever is lying around")
    return [name for name in out.stdout.split("\x00") if name]


def _commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _scan(label: str, text: str, markers: tuple[str, ...]) -> None:
    low = text.lower()
    for marker in markers:
        if marker in low:
            raise BuildError(f"{label} contains '{marker}': it would publish something from this machine. Nothing was written to the Space; clean or drop that file.")


def copy_energy_cache(source: Path, target: Path, markers: tuple[str, ...]) -> int:
    """A consistent copy of the energy cache at `target`, after checking every key and value. Returns the number of energies."""
    if not source.is_file():
        raise BuildError(f"no energy cache at {source}")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
        for key, payload in src.execute("SELECT key, payload FROM energies"):
            _scan("an energy in the cache", key + (payload if isinstance(payload, str) else payload.decode("utf-8", "replace")), markers)
        target.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(target)
        try:
            src.backup(dest)
            count = dest.execute("SELECT COUNT(*) FROM energies").fetchone()[0]
            dest.execute("VACUUM")
        finally:
            dest.close()
    return count


PREWARM = """# The full transfer study, computed here from the stored energies and the files in this image, so the first visitor is not the one who waits for it (and a missing energy fails the build).
COPY --chown=user prewarm.py ./prewarm.py
RUN python prewarm.py
"""


def dockerfile(host: str, prewarm: bool = True) -> str:
    """The Space's Dockerfile. The study is computed at build time only when there are stored energies to compute it from."""
    return f'''# syntax=docker/dockerfile:1
# Mission control, the substrate GUI, as a public demo. Built from {GITHUB} by deploy/huggingface/build_space.py.
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    SUBSTRATE_CACHE_DIR=/home/user/cache SUBSTRATE_MAX_ENSEMBLE={MAX_ENSEMBLE}
WORKDIR /home/user/app
COPY --chown=user pyproject.toml requirements-lock.txt ./
COPY --chown=user src ./src
RUN pip install --user -c requirements-lock.txt .
COPY --chown=user experiments ./experiments
COPY --chown=user cache /home/user/cache
{PREWARM if prewarm else ""}EXPOSE {PORT}
# The GUI is a local tool; this is the one place it listens beyond loopback, so it is told exactly which host name it answers to.
CMD ["python", "-m", "substrate", "gui", "--bind", "0.0.0.0", "--port", "{PORT}", "--no-browser", "--root", "/home/user/app", "--allow-host", "{host}", "--max-jobs", "{MAX_JOBS}"]
'''


def card(space_id: str, host: str, commit: str, energies: int, references: int) -> str:
    short = commit[:7]
    study = (f"""- Explore the **transfer study**: {references} real energy surfaces (from quantum chemistry), the result of fitting one cheap model to each and predicting the others. The full
  {references}-surface study is computed when the Space is built, so its matrix opens at once; any other selection is computed when you ask (it takes a while, and only {MAX_JOBS} jobs
  run at once).
""" if energies else "- **This build has no stored energies**, so only the model experiments work and there is no transfer study.\n")
    return f'''---
title: Substrate Mission Control
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: docker
app_port: {PORT}
pinned: false
license: other
short_description: Follow one process through five physical scales
---

# Substrate Mission Control

A demo of [substrate]({GITHUB}), a research prototype that follows one physical process through the scales at which it is modelled: from an electronic Hamiltonian,
through nuclear quantum dynamics and reaction kinetics, to an enzyme and the flux of a metabolic pathway. Nothing crosses a scale boundary except through a documented, validated
translator; if the physics does not support a hop, the run stops with the reason instead of inventing a result.

## What you can do here

- Pick an experiment, **edit any input**, and watch the whole chain re-run, stage by stage, with the approximations each stage made.
{study}
## What this demo cannot do

- **No new quantum chemistry.** PySCF is not installed. The real-chemistry experiments read {energies:,} energies that were computed with PySCF and stored (the energy cache);
  anything that needs an energy that is not stored fails with a message saying so. That includes the "downstream rates" panel, which runs the real chain at new distances.
- **It shares a small CPU with everyone.** At most {MAX_JOBS} jobs run at once (a third gets "the server is busy"), and an ensemble is limited to {MAX_ENSEMBLE} draws.
- **It is a research prototype.** The chemistry is a testbed on rigid, gas-phase scans, not a prediction of any real system's kinetics. Read the limits before quoting a number:
  [docs/limits.md]({GITHUB}/blob/main/docs/limits.md).

## Where it comes from

Source: [{GITHUB}]({GITHUB}), commit `{short}`. The Space is built from that commit by `deploy/huggingface/build_space.py`, with the energy cache copied from the author's
machine after a check that it holds nothing but numbers. Served at `https://{host}`.

License: none has been chosen yet, so the default copyright applies (all rights reserved); the card says `other`.

Quantum chemistry is computed with [PySCF](https://pyscf.org). Numerics use NumPy and SciPy.
'''


def build(out: Path, space_id: str, cache_dir: Path | None, extra_markers: tuple[str, ...] = ()) -> dict:
    host = space_host(space_id)
    if out.exists() and any(out.iterdir()):
        raise BuildError(f"{out} is not empty: choose a new folder (nothing is overwritten)")
    markers = machine_markers(extra_markers)
    files = _tracked()
    if not any(f.startswith("src/substrate/gui/static/") for f in files) or "requirements-lock.txt" not in files:
        raise BuildError("the git checkout is missing files the GUI needs (src/substrate/gui/static, requirements-lock.txt)")
    out.mkdir(parents=True, exist_ok=True)
    try:
        for name in files:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        energies = 0
        (out / "cache").mkdir()
        if cache_dir is not None:
            energies = copy_energy_cache(cache_dir / "qc.sqlite3", out / "cache" / "qc.sqlite3", markers)
        else:
            (out / "cache" / "README.txt").write_text("no energy cache was supplied; only the model experiments work\n", encoding="utf-8")
    except BaseException:                                       # a failed build leaves nothing behind: the folder was empty or new when it started
        for child in list(out.iterdir()):
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        raise
    references = sum(1 for f in files if f.startswith("experiments/references/") and f.endswith(".yaml"))
    commit = _commit()
    (out / "Dockerfile").write_text(dockerfile(host, prewarm=cache_dir is not None), encoding="utf-8", newline="\n")
    (out / "README.md").write_text(card(space_id, host, commit, energies, references), encoding="utf-8", newline="\n")
    if cache_dir is not None:
        shutil.copyfile(Path(__file__).with_name("prewarm.py"), out / "prewarm.py")
    (out / "SOURCE_COMMIT").write_text(commit + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"space_id": space_id, "host": host, "commit": commit, "files": sum(1 for f in out.rglob("*") if f.is_file()), "bytes": size,
            "energies": energies, "references": references}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a Hugging Face Space folder for Mission control.")
    parser.add_argument("outdir", type=Path, help="a new, empty folder")
    parser.add_argument("--space-id", required=True, help="owner/name of the Space, e.g. Hastika06/substrate-mission-control")
    parser.add_argument("--cache-dir", type=Path, help="the folder holding qc.sqlite3 (default: none, so only the model experiments work and there is no study to compute)")
    parser.add_argument("--also-refuse", action="append", default=[], metavar="TEXT", help="another string that must not appear in anything shipped (repeatable)")
    args = parser.parse_args(argv)
    try:
        summary = build(args.outdir, args.space_id, args.cache_dir.expanduser() if args.cache_dir else None, tuple(args.also_refuse))
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
