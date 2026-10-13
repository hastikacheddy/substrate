"""Compute the full transfer study ahead of time, so that a visitor's first request for it is answered from a saved result.

It reads the stored energies (no quantum-chemistry program is needed or present), fits every reference and compares each with each, exactly as the GUI's "Run" does, and saves the result
where the GUI looks for it. It runs against the files it is pointed at, so the saved result's key matches them: a study copied from another machine goes stale as soon as one
reference file differs, and this cannot. If an energy that a reference needs is not stored, this fails and so does the build, which is the right place to find out.

    python prewarm.py                 inside the Space's image: the project is /home/user/app and the package is installed
    python prewarm.py STAGING         for the static Space: STAGING holds src/, experiments/ and cache/qc.sqlite3, and its own source and cache are used
"""
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("/home/user/app")
if len(sys.argv) > 1:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    os.environ["SUBSTRATE_CACHE_DIR"] = str(root / "cache")

from substrate.gui import App  # noqa: E402  (after the path and the cache folder are set)

started = time.time()
app = App(root)
names = [r["name"] for r in app.references_listing()]
if len(names) < 2:
    sys.exit("there are fewer than two references to compare")
print(f"computing the transfer study of {len(names)} references from the stored energies", flush=True)
result = app.run_transfer(names, log=lambda text: print("  " + text, flush=True))
if app.cached_transfer(names) is None:
    sys.exit("the study was computed but not saved where the GUI will look for it")
print(f"saved: {len(names)} references in {time.time() - started:.0f} s ({result.get('seconds')} s of fitting)", flush=True)
