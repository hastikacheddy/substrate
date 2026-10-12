# Adding the next scale or model

[← back to the README](../README.md)

1. Subclass `Engine` for the new scale or model kind (claim a `kind`).
2. Subclass `Translator` from the previous scale: state its `approximations`, set `target_kind` (and `source_kinds` if it only
   accepts some models), check the assumptions in `validate()`. Anything the new scale needs that no lower scale computes goes in
   as a `context.*` parameter.
3. Register both in `defaults.default_registry()`. The planner finds the path; the CLI and ensembles work unchanged.

The metabolic engine already takes arbitrary stoichiometry and rate laws, so a second pathway topology only needs a translator that
builds its `structure`. Another quantum-chemistry program (xtb, Psi4, ...) is one `QCProgram` subclass plus `register_program`; the engine,
the cache and everything downstream are unchanged. The natural next steps: a richer or physically constrained model form (the fitted parameters
transfer neither across molecules nor from Hartree–Fock to correlated methods, so a form whose parameters keep their meaning is the thing to try; separate donor and
acceptor diabats and a heavy-atom term that is a real potential are untested ideas); more derivatives of a symmetric parent (a larger alkyl group, another substituent, the same methyl on bifluoride) to see whether the 0.27–0.38 eV per eV of proton-affinity gap holds;
the systematic part of the calibration's error (the part parameter uncertainty cannot see); relaxed scans, in which the groups' internal coordinates follow the
proton; higher levels of theory and a basis-set study; a 2D nuclear quantum treatment so the two chemistry routes can be compared
properly; and regulation, expression and noise at the biological scale.

## Connecting more real data

`substrate.datasets` is the pattern for bringing in measured or curated numbers, and the NIST proton affinities and acidities are the first table in it.
A new dataset is four small pieces: (1) a YAML table in `src/substrate/data/` in which every number carries its quantity, unit, uncertainty (or an explicit
`null` where the source gives none), method and citation, plus a `source` block saying where and when it was retrieved; (2) a loader that validates it
(`load_reference_set` refuses duplicate ids and species it does not know) and converts units in one place; (3) a rule that says which of the numbers
a computed quantity should be compared with, and refuses combinations that are not the same kind of thing (a proton affinity is not an acidity);
(4) a script that puts the computed and measured values side by side and reports the difference, including every approximation between them (here: rigid
against relaxed fragments, and an electronic energy against an enthalpy at 298 K), so a disagreement can be traced to a cause.

What was learned doing it, which the next dataset should expect: the first comparison is nearly always wrong for a reason in the *comparison* (a
rigid scan is not a separated, relaxed molecule; an electronic energy is not an enthalpy), not in the chemistry, and the corrections can be as large as the
effect being tested (0.18 eV for one ion here). Candidates, in the order of how directly they fit the existing machinery: more NIST WebBook quantities (ionisation
energies, electron affinities, heats of formation) for the same fragments; the GMTKN55 reaction-energy benchmark subsets for hydrogen bonds and proton
transfer, which would put a number on each level of theory and basis used here; JPL, KIDA or UMIST rate tables for the reaction scale; and BRENDA or SBML models
for the metabolic scale. None of these is built.

## Adding to the control plane

**A data source for the data fabric** is an `Adapter` subclass in `control/fabric/adapters.py`: an `id`, a `source` and `license` (both shown to users and stored with every dataset
version), a `hosts` property naming the only hosts it may contact (empty for local data), `check_query` (refuse anything that is not a short, plain value; the query goes into a URL),
`dataset_id` (the stable name a query maps to, so a repeated fetch becomes the next version of the same dataset), `url`, `parse` (bytes in, a list of records out) and `fields`, a tuple of `Field(name, type, ...)` that every record must satisfy (type, range, pattern, length) before anything is stored.
Register it in `default_adapters()`. The client enforces the allowlist, HTTPS, the size cap and the address checks; an adapter does not, and cannot, switch any of that off. Test it against a
local server that returns what the source's documentation says (see `tests/test_control_fabric.py`), and run it once against the real service before trusting it: a documented shape and a real one
differ more often than you would like.

**A new kind of task** is a handler in `control/tasks.py` (`handler(context) -> (output, lineage)`, the output being JSON-able and stored as an artifact), a `TaskSpec` where the workflow is
planned, and a rule in `control/status.py` for which status axis a failure of that kind belongs to. Decide whether its refusals are deterministic (never retried) in
`status.is_deterministic_refusal`. If it needs a different kind of worker, give it a queue name and start workers with `--queues`.

**A quantum provider** is a backend class with the `SolverBackend` interface, added to `qpu.backend_from_config`, and its name in `QuantumPolicy.providers` and `hardware_providers`. The gateway then
checks the policy and spends the submitter's `qpu_jobs_per_day` allowance for each task, and writes an audit event either way. The credentials belong to the gateway's process and must be read
from the environment or a secrets file there; never from an experiment, and never in the worker's scrubbed environment (the sandbox removes `IBM_`, `QISKIT_`, `AWS_` and similar names).

**A new role or permission** is an `Action` in `control/security.py`, an entry in `PERMISSIONS`, and `Depends(allowed(Action.X))` on the endpoint. `tests/test_control_registry.py` pins the table, so a change
to who may do what is a decision that has to be made twice.
