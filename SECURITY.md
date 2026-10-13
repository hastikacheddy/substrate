# Security

## Reporting a vulnerability

Please report a suspected vulnerability privately, through GitHub: the repository's **Security** tab, then **Report a vulnerability**. Please do not open
a public issue for it. Include what you ran, what you expected and what happened; a minimal experiment file is ideal.

## What this project is, and the trust model it assumes

`substrate` has two ways of running, with different threat models.

**1. A research tool on one user's own machine** (`python -m substrate run`, `gui`, `calibrate`, `transfer`). It reads experiment files, runs scientific models (and,
optionally, a quantum-chemistry program), caches the results on disk, and serves a local web page. There is no login, no tenant separation and no remote access, and none is claimed.

**2. A control plane for several people** (`python -m substrate platform ...`, the `platform` extra; [docs/platform.md](docs/platform.md)). An API, a database, workers, an artifact
store, and optional connections to public data sources. This is where authentication, authorisation, tenant isolation, quotas and an audit trail exist, and where the
threat model is a user who is authenticated but not trusted: one tenant's researcher must not be able to see, change or learn of another's work, exhaust the shared service,
or reach what the service can reach.

| Boundary | Trusted? | Why it matters |
|---|---|---|
| **Experiment files** (YAML or JSON), including ones from a colleague or a paper's supplementary data | **No** | The most likely route for a bad input to reach the program. |
| **Web pages open in the same browser** as the local GUI | **No** | A page can make the browser send requests to `127.0.0.1`. |
| **Numbers inside those files** (grid sizes, sample counts, parameters) | **No** | A single number can turn a small run into an enormous one. |
| **The energy cache and the data tables on disk** | Mostly | They are the user's own files, but they can be damaged or left behind by a buggy version. |
| **Python dependencies** (NumPy, SciPy, PyYAML, and PySCF if used) | At the pinned versions | Third-party code runs with the user's privileges. |
| **The user, and anything running as the user** | **Yes** | See "Not defended against". |
| *Control plane:* **an authenticated API client** | **No** | A tenant's user may try to read another tenant's objects, exceed their allowance, or send hostile input through a valid key. |
| *Control plane:* **the network between client and API** | **No** | The API speaks plain HTTP and expects a TLS-terminating proxy in front. |
| *Control plane:* **data from external sources** (PubChem, RCSB, NASA JPL) | **No** | The content, the redirects, the DNS answers and the size of what comes back are all outside our control. |
| *Control plane:* **the workers, the database and the artifact store** | **Yes** | A worker holds the database credential. Anything that compromises a worker compromises every tenant (see "Not defended against"). |
| *Control plane:* **the deployment's secrets and images** | **Yes** | Supplied by whoever deploys it. |

## What is in place, and where it is tested

### Both modes

| Threat | Control | Code | Tests |
|---|---|---|---|
| An experiment file runs code | Experiments are data. YAML is read with `safe_load` only; the package has no `eval`, `exec`, `pickle`, `marshal`, `os.system` or `shell=True`, and a static check enforces it. The engines a file can name are a fixed registry. | `safeload.py`, `defaults.py` | `test_policy.py` |
| A tiny file expands to billions of nodes (YAML aliases), or nests until the stack overflows, or is simply huge | Aliases are refused (the event stream is scanned before anything is built), nesting is capped at 40 levels, nodes at 200,000, file size at 1 MB. JSON `NaN` and `Infinity` are refused. | `safeload.py` | `test_safeload.py` |
| A number asks for an enormous computation | Every count (grid points, time steps, eigenvector entries, ensemble draws, quantum-chemistry jobs, expensive jobs, atoms per job, references in a study) has a limit, checked **before** anything is allocated or computed. The limits come from environment variables (`SUBSTRATE_MAX_...`), never from the experiment file, so a file cannot raise its own ceiling. | `limits.py`, the engines, `qc/`, `pipeline.py` | `test_limits.py` |
| A number is not a number (`nan`, `inf`, `2.5` samples, a negative sigma, a boolean count) | Checked when the file is parsed and again where the number is used. | `experiment.py`, `limits.py` | `test_limits.py` |
| A name from the file reaches the quantum-chemistry program as a file path | Theory and basis names must look like names (letters, digits and a few symbols; no path separators, dots or spaces). PySCF would otherwise treat a basis string as a file to open. | `qc/__init__.py` | `test_limits.py` |
| A web page in the user's browser drives the local GUI | The server binds to `127.0.0.1` only, refuses any request whose `Host` or `Origin` is not itself (which also blocks DNS rebinding), caps request bodies at 1 MB, confines file paths to `experiments/`, rejects non-finite numbers, and limits queued jobs. | `gui/server.py`, `gui/app.py` | `test_gui.py` |
| Command injection through the quantum-chemistry worker | The only process the package starts for chemistry is the PySCF worker. Its command is built from fixed parts with `shlex.quote`; job data travels as JSON on standard input, never in the command line. The static check allows only the listed modules to start a process or open a socket. | `qc/__init__.py` | `test_policy.py`, `test_qc.py` |
| A stale, damaged or hand-edited energy is served as if it had just been computed | Each cache entry is sealed with a SHA-256 digest and records the worker that wrote it. A failed digest or a non-finite energy is never served and is recomputed. `SUBSTRATE_QC_CACHE_STRICT=1` serves only entries written by the current worker. `python -m substrate cache status / verify / purge` audits and cleans it. | `qc/__init__.py`, `cli.py` | `test_qc_cache.py` |
| A result cannot be traced to its inputs | Every product records its inputs, backend, approximations, what its translator discarded, and warnings, and each hop's input hash is the previous hop's output hash. | `ir.py`, `pipeline.py` | `test_pipeline.py`, `test_discarded.py` |
| Vulnerable or tampered dependencies | Runtime and control-plane versions are pinned (`requirements-lock.txt`, `requirements-platform-lock.txt`); CI audits both with `pip-audit`, writes CycloneDX software bills of materials, runs CodeQL, builds the container image and starts it under the deployment's restrictions, and reviews dependency changes on pull requests; Dependabot proposes updates to the Python, Docker and Compose dependencies and the pinned Actions; GitHub Actions are pinned to commit hashes and run with read-only repository permissions (the CodeQL job also needs to write its security alerts). | `.github/` | the workflows; `test_deploy.py` checks their shape |

### The control plane

| Threat | Control | Code | Tests |
|---|---|---|---|
| A stolen or guessed API key | Keys are 256 random bits, shown once, and stored only as the SHA-256 of the secret. The check is constant-time, costs the same for an unknown key id as for a wrong secret, and says only "invalid" for unknown, malformed, wrong, revoked and disabled alike. Keys can be revoked and principals disabled; the credential travels in the `Authorization` header only (a key in a URL or a cookie is not a credential). | `security.py`, `registry.py`, `api.py` | `test_control_registry.py`, `test_control_api.py` |
| One tenant reads or learns of another's data | Every query for user data filters by the principal's tenant, and a lookup of another tenant's experiment, run, artifact or event answers "not found", the same as for something that does not exist. Artifacts are stored under the tenant's own namespace and are only readable through a registry record the tenant owns. | `registry.py`, `artifacts.py` | `test_control_registry.py`, `test_control_api.py`, `test_control_artifacts.py` |
| A user does something their role does not allow | Five roles, each with an explicit set of actions (a viewer reads; a researcher defines and runs; an operator watches, cancels and reads metrics and the audit trail but never the data; an administrator manages the tenant; a service credential does one thing). Every endpoint declares the action it needs. | `security.py`, `api.py` | `test_control_registry.py`, `test_control_api.py` |
| One user exhausts the service | Per-principal daily allowances (runs, tasks, compute-seconds, dataset fetches, quantum jobs) and per-tenant ceilings (concurrent runs, stored bytes, experiments), checked before work is queued and charged back if the request is refused; every limit in `limits.py` still applies inside a step; bodies over 1 MB are refused with 413 whether or not they declare their size. | `registry.py`, `orchestrator.py`, `worker.py`, `api.py` | `test_control_registry.py`, `test_control_orchestrator.py`, `test_control_api.py` |
| A malformed or hostile request | Request models forbid unknown fields and bound every length and range; unexpected errors answer "internal error" and log the details server-side; responses carry `nosniff`, `no-store`, a content-security policy that allows nothing and `X-Frame-Options: DENY`; artifacts are served with their stored type and as attachments unless they are JSON. | `api.py` | `test_control_api.py` |
| A result or an audit record is altered afterwards | The audit trail and each run's lineage are hash chains: every event commits to the one before it, a unique index stops two events from taking the same place, and the API reports whether a chain verifies and where it first does not. | `registry.py` | `test_control_registry.py` |
| A stored artifact is swapped or damaged | Artifacts are named by the SHA-256 of their bytes, and every read recomputes it. A narrow set of media types is accepted (JSON without `NaN`, text, CSV, `.npy` loaded with `allow_pickle=False`), a size ceiling applies, and nothing is ever unpickled. | `artifacts.py` | `test_control_artifacts.py` |
| A step takes the worker down, runs forever, or reads what it should not | With `--isolation process` each step runs in its own spawned process with a deadline after which it is killed, an environment stripped of anything that looks like a credential (and of the database and S3 settings), and (on Linux) address-space, CPU-time and file-size limits. The parent reads the child's answer as bytes and parses JSON; it never unpickles. | `sandbox.py` | `test_control_sandbox.py` |
| A data adapter is made to reach something it should not (SSRF) | The only code that fetches data on a user's behalf is one client (the database, Redis, S3, the telemetry collector and IBM's service are connections the deployment configures, not ones a request can aim): allow-listed hosts, HTTPS, no credentials or odd ports or fragments in the URL, the name resolved once and every private, loopback, link-local (including the cloud metadata address), multicast, reserved and shared-address result refused, the connection made to the vetted address itself (so DNS rebinding cannot redirect it), redirects re-checked within the allowlist and limited to three, status 200 and a listed content type only, a byte cap, and no query string in any error message. Adapters then validate every field of every record. | `fabric/client.py`, `fabric/adapters.py` | `test_control_fabric_client.py`, `test_control_fabric.py` |
| Quantum compute is used beyond what is allowed | A policy bounds provider, qubits, depth, gates, shots and circuits and refuses before anything runs; a hardware provider would also spend a per-user allowance that is zero until an administrator grants some; every refusal is an audit event. The providers a deployment allows are named in `SUBSTRATE_QPU_PROVIDERS` (default: the simulator only). The Qiskit providers (`qpu_qiskit.py`) are checked against the same policy before anything is built or sent; no real device has been run. | `qpu.py`, `qpu_qiskit.py`, `gateway.py` | `test_qpu.py`, `test_qpu_qiskit.py`, `test_control_gateway.py` |
| A quantum-provider token leaks, or reaches a step | The IBM token is read only from `QISKIT_IBM_TOKEN` or the file named by `QISKIT_IBM_TOKEN_FILE`; it is not in any message (an error from the service is reduced to its type name, and the original is suppressed), and a step in a sandbox has a scrubbed environment, so it never receives it: hardware is available only with `--isolation inline`, where the worker process itself holds the token. | `qpu_qiskit.py`, `sandbox.py` | `test_qpu_qiskit.py` (against a stand-in service) |
| The wake-up channel is forged, flooded or down | A Redis notification carries no data: the worst a forger can do is make a worker query the database sooner. The queue and every decision stay in the database. Redis I/O is on background threads, so an outage does not slow the API or a worker (measured: submit 0.1 s with Redis stopped, 4 to 8 s before that was fixed). The Redis password is a secret file, and the URL never appears in a log or an error. | `notify.py` | `test_control_notify.py` (fake client and a real Redis) |
| A secret leaks into a log, an error or a metric | The database URL comes from a file (`SUBSTRATE_DB_FILE`), not the environment; API keys are read from the environment, never the command line; log records are redacted (tokens, `Authorization` headers, `password=...`) before they are written; error messages from the fetch client never contain a query string; metric labels have a handful of values and never name a run, a task or a person. | `cli.py`, `observability.py`, `fabric/client.py` | `test_control_observability.py`, `test_control_fabric_client.py`, `test_control_cli.py` |
| A container is a foothold | The image runs as an unprivileged user from pinned wheels; Compose and Kubernetes run it with a read-only root filesystem, no capabilities, no privilege escalation and (Kubernetes) the restricted Pod Security Standard and no service-account token; ports are published on loopback only; the database, workers and telemetry sit on internal networks, the workers able to reach the database and the collector and nothing else; Kubernetes network policies deny by default and keep the API's way out away from private, link-local and metadata addresses; secrets are mounted as files. | `deploy/` | `test_deploy.py` (structure and properties; see [docs/platform.md](docs/platform.md) for what has not been run) |

## Not defended against

- **Anything running with the user's privileges** (the local tool). A malicious local process can edit the cache (and re-seal it), the source, or the lock files. The digest on
  cache entries detects damage and accidental overwrites, not an attacker who can recompute it.
- **Untrusted Python.** Experiments are data, but `register_program` and `register_backend` take Python callables, and installing a package runs its code.
  Do not load a plugin you have not read.
- **A compromised worker.** Workers connect to the database directly, with the database credential, so a worker that is compromised can read and change every tenant's
  data and forge audit records. The process sandbox protects a worker from a *crashing or runaway step*, not from hostile code (the engines are a fixed registry and the experiment is
  data, so the step runs code this package ships); the container is the boundary, and per-worker database roles or a broker that hands workers only their own tasks would be the next step.
- **The audit chain against someone who controls the database.** It makes any edit, deletion or reordering detectable by a reader who has a trusted copy of the latest hash; it does
  not stop an administrator of the database from rewriting the whole chain consistently. There is no external anchor or signature.
- **No TLS.** The API speaks plain HTTP and refuses to listen beyond loopback unless told to; put a TLS-terminating proxy in front.
- **No rate limiting by address.** Allowances are per principal per day. An unauthenticated flood costs the service one database lookup per request.
- **No single sign-on, no multi-factor, no key expiry.** Keys last until revoked. An OIDC-validating proxy could map claims to a principal, which is why nothing below the API cares
  how a principal was established, but nothing here does it.
- **Confidentiality at rest.** Nothing is encrypted by this software: not the database, not the artifacts, not the cache. Use the storage's encryption. Do not put secrets in an experiment.
- **Shared datasets.** Ingested datasets are public data and are deliberately visible to every tenant; the only bound on how many a user can add is the daily ingest allowance.
- **Provenance is not authenticity.** The hash chain shows that a result is consistent with its recorded inputs; it is unsigned, and the identifiers are truncated for readability.
- **Hash-pinned installs and signed images.** The lock files pin versions, not hashes (a hash is per wheel and per platform); the image is not signed or scanned by this project, and its base is pinned by tag unless
  you pin it by digest.
- **The quantum-chemistry program itself.** PySCF is trusted code; its own parsing of the theory and basis names is only guarded by the name check above.
- **Anything about the deployment files beyond what was run.** They were run under Docker Desktop on one Windows machine (the stack, with a run submitted and reproduced) and on a one-node
  minikube cluster with Calico (both the manifests and the Helm chart; 13 network-policy checks behaved as designed). Not run: a Linux host, a managed cloud cluster, more than one node, another
  network plugin, an ingress with TLS. A network plugin that does not enforce NetworkPolicy makes those objects decorative. See [docs/platform.md](docs/platform.md).
- **A hardware provider's token in the worker.** With `--isolation inline` the worker process holds the IBM token, so anything that compromises that worker has it; it is as trusted as the database credential it already holds.

## Before exposing it beyond a trusted network

The pieces above are what a validated specification of a multi-user service needs first. What it still needs: TLS and a request rate limit in front of the API; a credential per worker
(or a broker) so that a worker does not hold the whole database; a secrets manager instead of mounted files; single sign-on and key expiry; an external anchor for the audit chain;
image scanning and signing; encryption at rest; backups and a restore test; a hardware quantum provider behind the gateway, with credentials the gateway alone holds; and an independent
review of all of it. None of that is claimed to exist.
