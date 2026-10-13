# Mission control as a Hugging Face Space

There are two ways to put Mission control on Hugging Face, and the difference is who pays and where the Python runs:

| | runs the Python | free account | built by |
|---|---|---|---|
| **Static Space** (below, "in the browser") | in the visitor's browser (Pyodide, WebAssembly) | yes | `build_static_space.py` |
| **Docker Space** (the rest of this file) | on Hugging Face's server (the local GUI in its public mode) | **no: needs PRO** | `build_space.py` |

`upload_space.py` puts either folder on Hugging Face (private first). Nothing here is part of the package: it is the recipe for one deployment, kept so that the Space can be rebuilt from a commit.

## A static Space: the demo, in the visitor's browser

```bash
python deploy/huggingface/build_static_space.py ../space --space-id Hastika06/substrate-mission-control --cache-dir ~/.cache/substrate
python deploy/huggingface/upload_space.py ../space --space-id Hastika06/substrate-mission-control       # private; --public is a separate, later step
```

A static Space only serves files, so there is no server to run the GUI on. Instead the page is the GUI's own, and a small shim (`static_space/web_shim.js`) answers its `/api/...` calls from a web
worker (`static_space/worker.js`) running [Pyodide](https://pyodide.org) (CPython compiled to WebAssembly; version pinned, loaded from the jsDelivr CDN) with numpy, scipy and PyYAML. The worker
unpacks `bundle.zip` (the package source, the experiments, the energy cache and the transfer study) and answers with `substrate.gui.api`, the same code the local server runs. A job runs to its
end inside the request that started it, because a worker has no threads (`App(inline_jobs=True)`), so the page shows a job's log only once it is over.

- The 24-surface study is computed by the builder, natively, from the very files in the bundle (a saved study is keyed to their bytes, and the test `test_the_study_in_the_bundle_is_the_one_the_gui_looks_up_for_the_bundles_own_files` holds that), so its matrix opens at once.
- Anything else is computed in the visitor's browser: about 25 seconds for eight surfaces on the machine it was tried on.
- **Numbers recomputed in the browser can differ from native ones.** Pyodide's numpy/scipy are not the versions the project is locked to. Compared with a native run: one real-chemistry experiment, identical
  (237 of 237 numbers); a three-surface study, median relative difference 8e-8 and 90% under 1e-5, but 10 of 3,393 numbers off by more than 1e-3 and a few near-threshold outcomes flipped (a second well or a barrier found
  or not, on the bifluoride surface). The card says so.
- It needs a browser with WebAssembly and web workers, downloads about 40 MB on the first visit, and fetches Pyodide from a CDN, so it does not run offline.
- **On a private Space the page's requests are authorised and a worker's are not.** The Space's page is loaded with a signed `?__sign=` address; the page and the scripts it loads work, but a `fetch` made
  from inside the worker was answered 401 (seen on Hugging Face on 2026-10-14, after everything passed locally). So the page downloads `bundle.zip` and hands the bytes to the worker.
  The page shows "not computed yet" for the transfer study until Python has started, because the worker answers one request at a time and Python takes about half a minute to load.

## A Docker Space: the local GUI as a server

`build_space.py` assembles a Docker Space that runs the local GUI (`python -m substrate gui`) as a public demo.

```bash
# 1. a folder holding exactly what the Space ships (it refuses a folder that is not empty)
python deploy/huggingface/build_space.py ../space --space-id Hastika06/substrate-mission-control --cache-dir ~/.cache/substrate

# 2. log in once, in your own terminal (the token is typed there; this repository's scripts never read or print one)
hf auth login

# 3. create the Space (private) and upload; add --public later, as a separate step, when you have looked at it
python deploy/huggingface/upload_space.py ../space --space-id Hastika06/substrate-mission-control
```

## What the Space is

- **A snapshot of a commit.** The files are the ones git tracks under `src/`, `experiments/` and the two lock/metadata files, and the card names the commit. A Space built from a dirty checkout
  would not match the commit the card names, so build from a clean one.
- **The GUI in its opt-in public mode.** The GUI is a local tool: by default it listens on 127.0.0.1 and refuses any Host or Origin that is not `127.0.0.1` or `localhost`. The Space starts it with
  `--bind 0.0.0.0 --allow-host <the Space's own host name> --max-jobs 2`: it listens beyond the machine only because it was told the one name it answers to (no wildcards, and it refuses to
  start without one), the Host and Origin checks stay on for that name, and at most two jobs run at once (a third gets HTTP 429, "the server is busy"). `SUBSTRATE_MAX_ENSEMBLE=200` keeps an
  ensemble small. The container runs as an ordinary user, with no secrets in it.
- **The stored energies, not new chemistry.** PySCF is not installed. The real-chemistry experiments read the energy cache (`qc.sqlite3`, thousands of PySCF energies, numbers only) from
  `--cache-dir`; the builder copies it with SQLite's backup API and **stops if any value contains a user name, a home or drive path or a host name**, leaving nothing behind. Anything that
  needs an energy that is not stored fails with a message saying so. The "downstream rates" panel is the one place this shows up in a finished job: it runs the real chain at the distance where a
  barrier height is reached, which is stored for the default 0.4 eV on most surfaces (23 of 24 when checked) and for few at 0.1, 0.9 or 1.3 eV, where a cell shows a dash and its tooltip says why.
- **The transfer study is computed in the image.** Saved studies are not copied from your machine: a study is keyed to the exact bytes of the reference files it was made from, so one made
  here goes stale as soon as any of them changes (all seven of the author's did). `prewarm.py` runs the full study from the stored energies while the image is built, which also makes
  the build fail, rather than the Space, if an energy is missing. With no `--cache-dir` there is nothing to compute it from, so the step is left out.
- **License unspecified.** The card says `license: other`, matching GitHub, where none has been chosen.

## Things to know

- **A Docker Space on free hardware needs a Hugging Face PRO subscription.** On 2026-10-13 creating one from a free account was refused with `402 Payment Required` ("Static Spaces are free for
  everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"). The uploader reports that refusal and stops; nothing is created. The GUI is a Python server, so a
  static Space cannot stand in for it.

- A **private** Space is visible only to its owner. Making it public is `upload_space.py ... --public`, or the switch in the Space's settings; copies and search indexes can outlive a later deletion.
- The Space is served from `https://<owner>-<name>.hf.space` (lower-case, dots and underscores turned into hyphens); that is the host name the GUI is told to accept. If Hugging Face's own health
  checks or proxy use another Host header, the GUI answers 403 to them: the Space's logs say so, and `--allow-host` takes more than one name.
- Free Space hardware is a small shared CPU that sleeps when idle; the first request after a sleep waits for it to wake.
