# Security

## Reporting a vulnerability

Please report a suspected vulnerability privately, through GitHub: the repository's **Security** tab, then **Report a vulnerability**. Please do not open
a public issue for it. Include what you ran, what you expected and what happened; a minimal experiment file is ideal.

## What this project is, and the trust model it assumes

`substrate` is a research tool that runs on **one user's own machine**. It reads experiment files, runs scientific models (and, optionally, a
quantum-chemistry program), caches the results on disk, and serves a local web page. It is not a multi-user service: there is no login, no tenant
separation and no remote access, and none is claimed. The threat model below is for that setting.

| Boundary | Trusted? | Why it matters |
|---|---|---|
| **Experiment files** (YAML or JSON), including ones from a colleague or a paper's supplementary data | **No** | The most likely route for a bad input to reach the program. |
| **Web pages open in the same browser** as the local GUI | **No** | A page can make the browser send requests to `127.0.0.1`. |
| **Numbers inside those files** (grid sizes, sample counts, parameters) | **No** | A single number can turn a small run into an enormous one. |
| **The energy cache and the data tables on disk** | Mostly | They are the user's own files, but they can be damaged or left behind by a buggy version. |
| **Python dependencies** (NumPy, SciPy, PyYAML, and PySCF if used) | At the pinned versions | Third-party code runs with the user's privileges. |
| **The user, and anything running as the user** | **Yes** | See "Not defended against". |

## What is in place, and where it is tested

| Threat | Control | Code | Tests |
|---|---|---|---|
| An experiment file runs code | Experiments are data. YAML is read with `safe_load` only; the package has no `eval`, `exec`, `pickle`, `marshal`, `os.system` or `shell=True`, and a static check enforces it. The engines a file can name are a fixed registry. | `safeload.py`, `defaults.py` | `test_policy.py` |
| A tiny file expands to billions of nodes (YAML aliases), or nests until the stack overflows, or is simply huge | Aliases are refused (the event stream is scanned before anything is built), nesting is capped at 40 levels, nodes at 200,000, file size at 1 MB. JSON `NaN` and `Infinity` are refused. | `safeload.py` | `test_safeload.py` |
| A number asks for an enormous computation | Every count (grid points, time steps, eigenvector entries, ensemble draws, quantum-chemistry jobs, expensive jobs, atoms per job, references in a study) has a limit, checked **before** anything is allocated or computed. The limits come from environment variables (`SUBSTRATE_MAX_...`), never from the experiment file, so a file cannot raise its own ceiling. | `limits.py`, the engines, `qc/`, `pipeline.py` | `test_limits.py` |
| A number is not a number (`nan`, `inf`, `2.5` samples, a negative sigma, a boolean count) | Checked when the file is parsed and again where the number is used. | `experiment.py`, `limits.py` | `test_limits.py` |
| A name from the file reaches the quantum-chemistry program as a file path | Theory and basis names must look like names (letters, digits and a few symbols; no path separators, dots or spaces). PySCF would otherwise treat a basis string as a file to open. | `qc/__init__.py` | `test_limits.py` |
| A web page in the user's browser drives the local GUI | The server binds to `127.0.0.1` only, refuses any request whose `Host` or `Origin` is not itself (which also blocks DNS rebinding), caps request bodies at 1 MB, confines file paths to `experiments/`, rejects non-finite numbers, and limits queued jobs. | `gui/server.py`, `gui/app.py` | `test_gui.py` |
| Command injection through the quantum-chemistry worker | The only process the package starts is the PySCF worker. Its command is built from fixed parts with `shlex.quote`; job data travels as JSON on standard input, never in the command line. The static check allows no other module to start a process or open a socket. | `qc/__init__.py` | `test_policy.py`, `test_qc.py` |
| A stale, damaged or hand-edited energy is served as if it had just been computed | Each cache entry is sealed with a SHA-256 digest and records the worker that wrote it. A failed digest or a non-finite energy is never served and is recomputed. `SUBSTRATE_QC_CACHE_STRICT=1` serves only entries written by the current worker. `python -m substrate cache status / verify / purge` audits and cleans it. | `qc/__init__.py`, `cli.py` | `test_qc_cache.py` |
| A result cannot be traced to its inputs | Every product records its inputs, backend, approximations and warnings, and each hop's input hash is the previous hop's output hash. | `ir.py`, `pipeline.py` | `test_pipeline.py` |
| Vulnerable or tampered dependencies | Runtime versions are pinned (`requirements-lock.txt`); CI audits them with `pip-audit`, writes a CycloneDX software bill of materials, runs CodeQL, and reviews dependency changes on pull requests; Dependabot proposes updates; GitHub Actions are pinned to commit hashes and run with read-only repository permissions (the CodeQL job also needs to write its security alerts). | `.github/` | the workflows themselves |

## Not defended against

- **Anything running with the user's privileges.** A malicious local process can edit the cache (and re-seal it), the source, or the lock files. The digest on
  cache entries detects damage and accidental overwrites, not an attacker who can recompute it.
- **Untrusted Python.** Experiments are data, but `register_program` and `register_backend` take Python callables, and installing a package runs its code.
  Do not load a plugin you have not read.
- **Provenance is not authenticity.** The hash chain shows that a result is consistent with its recorded inputs; it is unsigned, and the identifiers are
  truncated for readability.
- **Hash-pinned installs.** The lock files pin versions, not hashes (a hash is per wheel and per platform).
- **The quantum-chemistry program itself.** PySCF is trusted code; its own parsing of the theory and basis names is only guarded by the name check above.
- **Confidentiality.** Nothing is encrypted at rest. Do not put secrets in an experiment file; none is needed.

## If it becomes a hosted service

None of the following exists. A multi-user deployment would need, at least: authentication and per-resource authorisation, per-tenant isolation of data and
compute, jobs run in a sandbox (non-root, read-only root filesystem, no network except an allowlist, no cloud-metadata endpoint), per-user quotas
enforced by a scheduler, secrets issued to workers by a broker and never placed in experiment files, a policy gateway in front of any quantum hardware
(circuit, shot and backend limits), an artifact store that validates type and size and never deserialises pickles, and an audit trail of security
events. The limits and checks above are the starting point for the "validated specification" stage of that design, not a substitute for the rest.
