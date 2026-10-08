# Architecture

[← back to the README](../README.md)

The diagrams (system context, scales and kinds, the lifecycle of a run, the core types, the quantum-chemistry bridge, the model, calibration and transfer, and the GUI) are in the
[Architecture section of the README](../README.md#architecture). This page is the prose behind them.

## Two routes from a surface to a rate

Two real, tested routes from an electronic surface to a rate, which **make different approximations and are not
interchangeable**:

| | molecular route | quantum route |
|---|---|---|
| heavy atoms (O···O) | free to relax: the relaxed barrier | frozen at one chosen distance |
| proton | classical, zero-point energy only | quantum: tunnelling included |
| rate | Eyring TST, no tunnelling correction | ν·P(E) summed over thermal levels |

Neither is the full answer, which needs the nuclear quantum problem in both coordinates (not built). **Do not divide one
route's rate by the other to get a "tunnelling factor"**: at the default parameters they differ by 4 orders of
magnitude, almost all of it from where the quantum route freezes the O···O distance (`python examples/molecular_vs_quantum.py`).
The planner enforces this: asking for any scale above `molecular` from the 2D electronic kind raises `AmbiguousPathError`
and makes you name the route.

- **`ir.py`: the Scientific IR.** `ScientificSystem` = scale, kind, `parameters`, `state`, `observables`,
  `structure` (non-numeric topology), `dynamics`, and `provenance`. Numbers are `Quantity(value, unit, sigma, source, band)`;
  units are *checked*, not converted. `fingerprint()` is a content hash used for lineage.
- **Context parameters.** A parameter named `context.<name>` belongs to the experiment, not to one scale (a substrate
  concentration, a binding rate, an enzyme abundance). The pipeline carries it across every translator unchanged, so the
  biological scale can use an enzyme abundance that was specified at the electronic scale without the seven stages in between knowing
  it exists. Context parameters take part in uncertainty ensembles like any input.
- **`base.py`: contracts and routing.** `Engine` solves a system of one scale and *kind*. `Translator` maps a solved
  system to an unsolved one, names the kind it produces (`target_kind`) and the kinds it accepts (`source_kinds`). The
  `Registry` keys engines by `(scale, kind)` and finds translator chains kind-aware. **If several distinct chains tie for
  shortest, it raises `AmbiguousPathError` instead of picking one**, because different routes are different physics.
- **`backends.py`: `SolverBackend`.** Engines ask a backend for eigenpairs (`lowest_eigenpairs` for the nuclear Hamiltonian,
  `symmetric_eigh` for electronic Hamiltonians) and ODE integration. Only `ClassicalBackend` ships; a Qiskit/GPU backend is a
  subclass plus `register_backend`. `symmetric_eigh` is where a VQE solver would plug in. (The biophysical engine is small dense
  linear algebra and does not use the backend.)
- **`qc/`: quantum-chemistry programs.** A second seam, for external codes that compute molecular energies. An engine describes
  geometries as `QCJob`s (atoms, charge, spin, theory, basis, and a task: a single point; a `relax` that minimises the energy over every atom with analytic gradients and returns the relaxed geometry; or a `thermo` that relaxes tightly and adds the harmonic zero-point energy and the enthalpy at 298 K, for Hartree–Fock and DFT); a `QCProgram` returns `QCResult`s. `PySCFProgram` runs PySCF
  in-process if it is installed, or through a standalone worker (`pyscf_worker.py`, which imports nothing from substrate) in WSL, over a
  JSON protocol, in chunks so a crash loses at most one chunk. Energies are cached in SQLite by a content hash (geometry to 1e-8 Å,
  method, basis, charge, spin, program). A missing program raises `QCError`, deliberately *not* a `ValidationError`: an ensemble drops
  invalid draws, but it must stop for an environment failure. Non-convergence is reported through the result and becomes a
  `ValidationError`.
- **`datasets.py` and `fragments.py`: measured reference values.** `datasets.py` loads curated tables of measured numbers
  (`data/nist_ion_energetics.yaml`: gas-phase proton affinities and acidities from the NIST Chemistry WebBook) as `ReferenceValue`s that keep their unit
  conversion, uncertainty, method and citation; `experimental_well_gap(template)` turns two of them into the measured energy difference between
  the two wells of an asymmetric complex, and refuses to combine a proton affinity with an acidity. `fragments.py` takes such a complex apart into its four
  separated fragments (each side with and without the proton) so a QC program can relax them or give them a thermochemical correction; together they let
  `examples/validate_against_nist.py` put a computed gap beside the measured one.
- **`gui/`: the local web GUI.** `app.py` is everything the GUI does without HTTP (list and run experiments, apply input edits, run long work as
  background jobs the page polls, assemble the transfer study and save it); `serialize.py` turns solved systems into JSON and decides which arrays
  are worth plotting at each scale (the server chooses what to show; the page only draws three generic plot types); `server.py` is a standard-library
  HTTP layer; `static/` is plain HTML, CSS and JavaScript with no build step and no external requests. The server binds to loopback, refuses a Host
  or Origin that is not itself (so another page in the browser, or a DNS-rebinding trick, cannot drive it), requires JSON for every POST, serves only
  the files in `static/` by exact name under a strict Content-Security-Policy, and resolves experiment paths so they cannot leave `experiments/`.
- **`molecules.py`: molecule templates.** An experiment may name a molecule (`molecule: {template: ammonium_dimer_cation}`) instead of listing
  atoms: `zundel_cation` (O–H–O), `ammonium_dimer_cation` (N–H–N), `bifluoride_anion` (F–H–F) and `water_ammonia_cation` (O–H–N), `methanol_water_cation` (O–H–O, with a rigid staggered methyl group) and `ammonia_methylamine_cation` (N–H–N, likewise), the last three **asymmetric**, so they must be scanned without the mirror shortcut (the engine refuses a false symmetry claim), each with documented frozen flank geometry. The frozen orientations are options (`twist`, and the methyl `rotor`): turning the water–ammonia ion's twist changes the energy by at most 0.5 meV, but the methanol–water ion's is not free: its alternative arrangements move the HF energy by 16–52 meV (the default is the lowest of the five tried), by at most 16 meV across the grid and by at most 7.5 meV in the left–right difference that sets the asymmetry. The ammonia–methylamine ion's flank twist is irrelevant (at most 1 meV) but its methyl orientation is not: an eclipsed methyl is 103–119 meV higher at HF, a 30° turn 52–59 meV, and the left–right difference moves by up to 13 meV; the staggered default is the lowest. The
  engine resolves the name and records the full atom list, with the template's name, on the solved system, so a result always says which molecule it was.
- **`transfer.py`: does a calibration carry to another reference?** Cross-prediction of every calibration against every reference, the real chain's
  rates against the same chain on each calibration, and few-shot learning curves with a Gaussian prior (see [science.md](science.md)).
- **Experiment files** can carry a non-numeric `structure:` (a molecule, a method, a mechanism) alongside numeric parameters.
- **`calibration.py`: fitting the model engines to a reference surface** (see [science.md](science.md)), and **joint parameter draws** in the ensemble: a
  system may carry `structure["parameter_covariance"]` (names, a covariance matrix, optional physical minimums); those parameters are then
  drawn together from a multivariate Gaussian, truncated at the minimums by redrawing the vector, instead of independently, so correlations
  from a calibration are respected and no run is wasted on a draw the model correctly rejects (a negative well depth).
- **`workflow.py` / `pipeline.py`.** An experiment compiles to a DAG of `solve`/`translate` tasks. Every product gets a
  `ProvenanceRecord`: input and output fingerprints, backend, the component's approximations, warnings, timing.
  `RunResult.save()` writes every intermediate system as JSON.
- **Uncertainty.** If root parameters carry a `sigma`, the *whole chain* is re-run per draw and the spread is attached to
  every derived parameter and observable. Scalars also get a 68% percentile `band`, and the report prints it instead of `±`
  when the spread rivals the value: for exponentially sensitive quantities like rate constants a standard deviation
  can exceed the mean (a `±` larger than a positive rate is meaningless). Draws that fail validation are counted and dropped.
