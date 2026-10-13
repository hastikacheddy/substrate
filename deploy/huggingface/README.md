# Mission control as a Hugging Face Space

`build_space.py` assembles a Docker Space that runs the local GUI (`python -m substrate gui`) as a public demo, and `upload_space.py` puts it on Hugging Face. Nothing here is part of the
package: it is the recipe for one deployment, kept so that the Space can be rebuilt from a commit.

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
- **The stored energies, not new chemistry.** PySCF is not installed. The real-chemistry experiments read the energy cache (`qc.sqlite3`, thousands of PySCF energies, numbers only) and the saved
  transfer-study results from `--cache-dir`; the builder copies them with SQLite's backup API and **stops if any value contains a user name, a home or drive path or a host name**, leaving nothing
  behind. Anything that needs an energy that is not stored fails with a message saying so.
- **License unspecified.** The card says `license: other`, matching GitHub, where none has been chosen.

## Things to know

- A **private** Space is visible only to its owner. Making it public is `upload_space.py ... --public`, or the switch in the Space's settings; copies and search indexes can outlive a later deletion.
- The Space is served from `https://<owner>-<name>.hf.space` (lower-case, dots and underscores turned into hyphens); that is the host name the GUI is told to accept. If Hugging Face's own health
  checks or proxy use another Host header, the GUI answers 403 to them: the Space's logs say so, and `--allow-host` takes more than one name.
- Free Space hardware is a small shared CPU that sleeps when idle; the first request after a sleep waits for it to wake.
