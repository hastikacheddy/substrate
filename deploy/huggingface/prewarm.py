"""Compute the full transfer study while the Space's image is built, so that a visitor's first request for it is answered from a saved result.

It reads the stored energies (no quantum-chemistry program is needed or present), fits every reference and compares each with each, exactly as the GUI's "Run" does, and saves the result
where the GUI looks for it. It runs against the files that are in the image, so the saved result's key matches them: a study copied from another machine goes stale as soon as one
reference file differs, and this cannot. If an energy that a reference needs is not stored, this fails and so does the build, which is the right place to find out.
"""
import sys
import time

from substrate.gui import App

started = time.time()
app = App("/home/user/app")
names = [r["name"] for r in app.references_listing()]
if len(names) < 2:
    sys.exit("there are fewer than two references to compare")
print(f"computing the transfer study of {len(names)} references from the stored energies", flush=True)
result = app.run_transfer(names, log=lambda text: print("  " + text, flush=True))
if app.cached_transfer(names) is None:
    sys.exit("the study was computed but not saved where the GUI will look for it")
print(f"saved: {len(names)} references in {time.time() - started:.0f} s ({result.get('seconds')} s of fitting)", flush=True)
