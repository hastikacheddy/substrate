"""Assemble a *static* Hugging Face Space that runs Mission control entirely in the visitor's browser.

    python deploy/huggingface/build_static_space.py OUTDIR --space-id Hastika06/substrate-mission-control --cache-dir ~/.cache/substrate

A Docker Space needs a Hugging Face PRO subscription; a static Space is free. A static Space serves files and cannot run a server, so the Python runs in the visitor's browser instead:
Pyodide (CPython compiled to WebAssembly, loaded from a pinned version on the jsDelivr CDN) with numpy, scipy and PyYAML, in a web worker. The page is the GUI's own, unchanged,
except that a small shim hands its `/api/...` calls to the worker, which answers them with the same code the local server uses (`substrate.gui.api`).

What goes in the folder, and nothing else:

* the GUI's page and scripts, as git has them at HEAD, plus `static/web_shim.js`, `worker.js` and the card (`README.md`, `sdk: static`, license "other");
* `bundle.zip`: the package source and the experiments as git has them at HEAD, the energy cache (copied with SQLite's backup API and **scanned first** for a user name, a drive or
  home path or a host name: the build stops if any is found) and the full transfer study, computed here, natively, from those very files (so the saved result's key matches them).

It refuses to write into a folder that is not empty. It never reads a token and does not talk to Hugging Face; `upload_space.py` does that.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import scipy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_space as common  # noqa: E402  (the shared checks and the energy-cache copy)

HERE = Path(__file__).resolve().parent
PYODIDE_VERSION = "0.28.3"
PYODIDE_NUMERICS = "numpy 2.2.5 and scipy 1.14.1"                       # what that Pyodide release ships (its pyodide-lock.json), shown on the card
CODE_PATHS = ("src", "experiments")
PAGE_FILES = ("app.css", "app.js", "charts.js")
SHIM_TAG = '<script src="/static/web_shim.js"></script>\n'


def page(index_html: str) -> str:
    """The GUI's index page with the shim loaded before the scripts that call the API."""
    marker = '<script src="/static/charts.js"></script>'
    if index_html.count(marker) != 1:
        raise common.BuildError("the GUI's index.html is not the page this builder expects (no single charts.js script tag)")
    return index_html.replace(marker, SHIM_TAG + marker)


def card(space_id: str, commit: str, energies: int, references: int, native: str = "") -> str:
    """`native` names the numpy and scipy the precomputed study was computed with (empty when it was not precomputed)."""
    study = (f"The full {references}-surface study was computed ahead of time (natively, with {native}), so its matrix opens at once"
             if native else "There is no precomputed study in this build, so every study is computed in your browser")
    return f'''---
title: Substrate Mission Control
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: other
short_description: Follow one process through five physical scales
---

# Substrate Mission Control

A demo of [substrate]({common.GITHUB}), a research prototype that follows one physical process through the scales at which it is modelled: from an electronic Hamiltonian,
through nuclear quantum dynamics and reaction kinetics, to an enzyme and the flux of a metabolic pathway. Nothing crosses a scale boundary except through a documented, validated
translator; if the physics does not support a hop, the run stops with the reason instead of inventing a result.

## How it runs

This page has no server. The Python runs **in your browser**: [Pyodide](https://pyodide.org) (Python compiled to WebAssembly) loads from the jsDelivr CDN, together with numpy, scipy and
PyYAML, and runs the project's own code in a web worker. What you edit and run stays on your machine. The first visit downloads about 40 MB and takes a while to start; later visits
use the browser's cache.

## What you can do here

- Pick an experiment, **edit any input**, and watch the whole chain re-run, stage by stage, with the approximations each stage made.
- Explore the **transfer study**: {references} real energy surfaces (from quantum chemistry), the result of fitting one cheap model to each and predicting the others. {study};
  any other selection is computed in your browser when you ask (eight surfaces took about 25 seconds on the machine it was tried on), and the page shows a job's progress only when it ends.

## What this demo cannot do

- **No new quantum chemistry.** PySCF cannot run here. The real-chemistry experiments read {energies:,} energies that were computed with PySCF and stored (the energy cache);
  anything that needs an energy that is not stored fails with a message saying so. The "downstream rates" panel runs the real chain at the distance where a barrier height is
  reached: it works where that distance is already stored (the default 0.4 eV, for most surfaces) and shows a dash, with the reason, where it is not.
- **Recomputing can give slightly different numbers.** The browser runs Pyodide {PYODIDE_VERSION}'s builds ({PYODIDE_NUMERICS}), not the versions the project is locked to. Measured against a
  native run: one real-chemistry experiment gave identical numbers (all 237); in a three-surface transfer study most numbers that differed did so by less than one part in 10^5, but 10 of
  3,393 differed by more than one part in 10^3, and a few near-threshold outcomes flipped (whether a second well or a barrier was found on the bifluoride surface). Treat a freshly
  recomputed study as indicative.
- **An ensemble is limited to 200 draws,** so a long run does not freeze your tab.
- **It is a research prototype.** The chemistry is a testbed on rigid, gas-phase scans, not a prediction of any real system's kinetics. Read the limits before quoting a number:
  [docs/limits.md]({common.GITHUB}/blob/main/docs/limits.md).

## Where it comes from

Source: [{common.GITHUB}]({common.GITHUB}), commit `{commit[:7]}`. The Space is built from that commit by `deploy/huggingface/build_static_space.py`, with the energy cache copied from the
author's machine after a check that it holds nothing but numbers.

License: none has been chosen yet, so the default copyright applies (all rights reserved); the card says `other`.

Quantum chemistry was computed with [PySCF](https://pyscf.org). Numerics use NumPy and SciPy.
'''


def _zip(stage: Path, target: Path) -> int:
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(stage.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))     # the same bytes give the same file
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())
            count += 1
    return count


def build(out: Path, space_id: str, cache_dir: Path, extra_markers: tuple[str, ...] = (), prewarm: bool = True) -> dict:
    common.space_host(space_id)                                        # validates the id
    if out.exists() and any(out.iterdir()):
        raise common.BuildError(f"{out} is not empty: choose a new folder (nothing is overwritten)")
    markers = common.machine_markers(extra_markers)
    files = common._tracked(CODE_PATHS + ("pyproject.toml",))
    code = [f for f in files if f.split("/")[0] in CODE_PATHS]
    if not any(f.startswith("src/substrate/gui/static/") for f in code):
        raise common.BuildError("the git checkout is missing files the GUI needs (src/substrate/gui/static)")
    out.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="substrate-static-") as tmp:
            stage = Path(tmp)
            for name in code:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(common.ROOT / name, target)
            energies = common.copy_energy_cache(cache_dir / "qc.sqlite3", stage / "cache" / "qc.sqlite3", markers)
            if prewarm:
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
                run = subprocess.run([sys.executable, str(HERE / "prewarm.py"), str(stage)], env=env, capture_output=True, text=True, check=False)
                if run.returncode != 0:
                    raise common.BuildError("computing the transfer study failed:\n" + (run.stdout + run.stderr)[-1500:])
                for study in sorted((stage / "cache" / "gui").glob("transfer-*.json")):
                    text = study.read_text(encoding="utf-8")
                    common._scan(study.name, text, markers)
                    json.loads(text)
            count = _zip(stage, out / "bundle.zip")
        (out / "static").mkdir()
        for name in PAGE_FILES:
            shutil.copyfile(common.ROOT / "src" / "substrate" / "gui" / "static" / name, out / "static" / name)
        shutil.copyfile(HERE / "static_space" / "web_shim.js", out / "static" / "web_shim.js")
        shutil.copyfile(HERE / "static_space" / "worker.js", out / "worker.js")
        index = (common.ROOT / "src" / "substrate" / "gui" / "static" / "index.html").read_text(encoding="utf-8")
        (out / "index.html").write_text(page(index), encoding="utf-8", newline="\n")
        references = sum(1 for f in code if f.startswith("experiments/references/") and f.endswith(".yaml"))
        commit = common._commit()
        (out / "README.md").write_text(card(space_id, commit, energies, references, f"numpy {np.__version__} and scipy {scipy.__version__}" if prewarm else ""),
                                       encoding="utf-8", newline="\n")
        (out / "SOURCE_COMMIT").write_text(commit + "\n", encoding="utf-8")
    except BaseException:                                              # a failed build leaves nothing behind: the folder was empty or new when it started
        for child in list(out.iterdir()):
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        raise
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"space_id": space_id, "commit": commit, "files": sum(1 for f in out.rglob("*") if f.is_file()), "bytes": size, "bundle_files": count,
            "energies": energies, "references": references, "pyodide": PYODIDE_VERSION, "study_precomputed": prewarm}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a static Hugging Face Space folder that runs Mission control in the browser.")
    parser.add_argument("outdir", type=Path, help="a new, empty folder")
    parser.add_argument("--space-id", required=True, help="owner/name of the Space, e.g. Hastika06/substrate-mission-control")
    parser.add_argument("--cache-dir", type=Path, required=True, help="the folder holding qc.sqlite3")
    parser.add_argument("--also-refuse", action="append", default=[], metavar="TEXT", help="another string that must not appear in anything shipped (repeatable)")
    args = parser.parse_args(argv)
    try:
        summary = build(args.outdir, args.space_id, args.cache_dir.expanduser(), tuple(args.also_refuse))
    except common.BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
