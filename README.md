<div align="center">

# substrate

**Propagate scientific models across scales, through one representation, with every hop documented and validated.**

![python](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)
![tests](https://img.shields.io/badge/tests-429-brightgreen)
![status](https://img.shields.io/badge/status-research%20prototype-orange)
![quantum chemistry](https://img.shields.io/badge/quantum%20chemistry-PySCF-6f42c1)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero-dark.png">
  <img alt="Mission control: one proton transfer followed from an electronic Hamiltonian to the flux of a metabolic pathway, five stages, with the pathway over time and the control of its flux" src="docs/images/hero-light.png" width="900">
</picture>

</div>

`substrate` follows one physical process through the scales at which it is modelled, from an electronic Hamiltonian, through nuclear quantum
dynamics and reaction kinetics, to an enzyme and the flux of a metabolic pathway. Nothing crosses a scale boundary except through a
**Translator**: a written list of approximations plus a `validate()` that checks them against the actual numbers. If the physics does not support
the hop, the pipeline stops with a reason instead of inventing a result.

On top of that sits a quantitative study built on **real quantum chemistry** (PySCF, from Hartree–Fock to CCSD(T)): can the parameters of a cheap
valence-bond model, fitted to one molecule at one level of theory, say anything about another? They mostly cannot, and the repository measures why.

> **Status: a research prototype.** The software is carefully tested and its statements about itself are measured. The chemistry is a *testbed*
> built on rigid, gas-phase scans, not a prediction of any real system's kinetics. [Read the limits](docs/limits.md) before you quote a number.

## Highlights

- **One representation.** A `ScientificSystem` carries numbers with units (checked, not converted), uncertainty, and a provenance hash chain; every product
  records its inputs, its backend, the approximations of the component that made it, and its warnings.
- **It refuses instead of guessing.** Two chains of equal length are different physics, so the planner raises `AmbiguousPathError` and makes you name
  the route; a barrier below the zero-point energy stops the strip at that stage, with the reason, and keeps everything that completed.
- **Real quantum chemistry, cached.** Single points from PySCF (HF, DFT, MP2, CCSD, CCSD(T)) on any molecule with a donor–proton–acceptor axis, stored on disk
  by content so every later run, and every draw of an uncertainty ensemble, is instant.
- **Calibration with honest diagnostics.** Fit the valence-bond model to a real surface and see which parameters sit on a bound, how well each is determined,
  and how the fit predicts a distance it never saw. The fit's uncertainty is reported as *not* the model's error.
- **A transfer study.** 21 real surfaces (7 molecules, 3 methods; three more built) with cross-prediction, downstream rates and isotope effects through the real chain,
  few-shot learning curves, a coupled-cluster benchmark, and a test of whether the finding survives pinning the bonds to the free diatomics.
- **Mission control.** A local web GUI over the same engines: edit any input and the whole chain re-runs, or explore the transfer study as a clickable matrix.
- **Tested hard.** 429 tests, closed-form and independent checks, and mutation testing of the numerical code. Bugs found in my own work are listed, not hidden.

## Screenshots

All of these are the real GUI (`python -m substrate gui`) running on the cached energies in this repository.

**A real surface through to a rate.** The Hartree–Fock/6-31G* energy surface of the Zundel cation H₅O₂⁺ (a proton shared between two water-like units), computed
by PySCF, and the double well along the proton coordinate that feeds the nuclear quantum dynamics. The panel below lists what the stage assumed.

![The ground-state surface E(x, R) of the Zundel cation and the potential along the proton coordinate](docs/images/real-surface.png)

**Do fitted parameters transfer?** Every calibration applied, unchanged, to every other surface: row = whose parameters, column = whose real energies; the diagonal
is each surface fitted to itself. Anything much worse than the diagonal is a failure to transfer, and the "constant guess" row is what predicting one number would score.

![The cross-prediction matrix for eight reference surfaces](docs/images/transfer-matrix.png)

Click a cell to see why. Here the bifluoride's parameters are asked to predict the chloride–hydrogen fluoride ion: the real energies (dots), the bifluoride's
prediction (blue) and the ion's own fit (dashed), the barrier at every distance, and the scores.

![A cell of the matrix opened: real energies against one surface's parameters and the target's own fit](docs/images/transfer-detail.png)

<details>
<summary>More views: own fits, fitted parameters, few-shot learning</summary>

**Own fits.** How well each calibration reproduces its own surface, the diagnostics that say where it does not, and which bond model was fitted.

![Own-fit table](docs/images/own-fits.png)

**Parameters.** The fitted values with their uncertainty. The bond length and width look alike across molecules; the coupling and the heavy-atom well, which decide the barrier, do not.
`= donor` marks a surface fitted with one Morse curve for both bonds.

![Fitted parameters of eight references](docs/images/parameters.png)

**Few-shot learning.** Fit a target from only k of its heavy-atom distances, with and without another surface's parameters as a Gaussian prior, and score the held-out distances.

![Held-out error against the number of distances used](docs/images/few-shot-learning.png)

</details>

## Quickstart

```bash
git clone https://github.com/hastikacheddy/substrate.git
cd substrate
pip install -e ".[dev]"

python -m substrate run experiments/proton_transfer_pathway.yaml     # five scales, nine stages
python -m substrate gui                                              # mission control, http://127.0.0.1:8765/
pytest -m "not qc"                                                   # 365 tests, no PySCF needed (1.5–5 min)
```

An experiment is a YAML file. Naming only the destination scale is enough when the route is unambiguous, and a `sigma` turns an input into a Monte-Carlo draw through the whole chain:

```yaml
experiment:
  phenomenon: pathway_flux_from_proton_transfer
  system:
    scale: electronic_structure
    kind: electronic.evb_two_state
    parameters:
      coupling: {value: 0.6, unit: eV}
      context.k_release:    {value: 1.0e4, unit: 1/s}                  # carried across every scale untouched
      context.enzyme_total: {value: 1.0e-6, unit: M, sigma: 2.0e-7}    # sigma => an ensemble
  propagation: [biological]
```

### Real quantum chemistry

Real energies need [PySCF](https://pyscf.org), which has no Windows build. On Linux or macOS `pip install pyscf` is enough. On Windows, `scripts/setup_qc_env.sh` creates an
isolated environment inside WSL (no `sudo`, nothing system-wide, everything in `~/.substrate-qc`; remove it with `rm -rf ~/.substrate-qc`):

```bash
wsl bash /mnt/c/<path to this project>/scripts/setup_qc_env.sh
```

The engine finds PySCF itself (in-process if importable, otherwise through WSL); `SUBSTRATE_QC_MODE`, `SUBSTRATE_QC_WSL_DISTRO` and `SUBSTRATE_QC_WSL_PYTHON` override discovery.
Energies are cached by content in `~/.cache/substrate` (or `SUBSTRATE_CACHE_DIR`): the first run of a scan costs minutes, every later run is instant. A coupled-cluster point costs about a
minute and a half in a large basis, so those jobs are sent to the worker in small chunks.

### Examples

| Command | What it shows |
|---|---|
| `python examples/zundel_real_chemistry.py` | A real Hartree–Fock surface, through to a rate and an isotope effect |
| `python examples/calibrate_against_real_chemistry.py` | The cheap model fitted to that surface, and how far its rates are from the real chain |
| `python examples/transfer_across_molecules_and_methods.py` | The 21-reference study: every comparison (computes ~4,500 energies the first time) |
| `python examples/anchoring_the_bonds.py` | Hold every bond at its free diatomic's value and refit all 24 surfaces |
| `python examples/benchmark_bifluoride.py` | HF, B3LYP, MP2 and CCSD(T) in two basis sets on one grid (`--figure` draws the plot) |
| `python examples/pathway_isotope_effect.py` | When an isotope effect survives from the molecule to the pathway flux |
| `python examples/enzyme_isotope_effects.py`, `molecular_vs_quantum.py`, `donor_acceptor_sweep.py` | Commitment regimes; the two routes to a rate; the barrier against heavy-atom distance |
| `python -m substrate calibrate experiments/zundel_hf.yaml --out calibrated.yaml` | Fit the model to a real surface and write an experiment that runs in milliseconds |
| `python -m substrate transfer experiments/references/*.yaml` | Cross-prediction of any set of reference surfaces from the command line |

## Architecture

Two views. The **high-level** diagrams show what the pieces are and how the scales connect. The **low-level** diagrams show what happens inside a run, the
core types, the quantum-chemistry bridge, the model itself, the studies built on it, and the GUI. File names are under `src/substrate/`; the text is in
[docs/architecture.md](docs/architecture.md) and [docs/science.md](docs/science.md).

### High level

#### System context

What goes in, what the core does with it, and what talks to what.

```mermaid
flowchart TB
    subgraph INPUTS["Inputs"]
        YAML["Experiment file (YAML)<br/>system, propagation, ensemble,<br/>calibration hints"]
        MOL["Molecule templates<br/>Zundel, bifluoride, chloride-HF, ..."]
    end

    subgraph CORE["Core"]
        EXP["experiment.py<br/>parse and validate"]
        PIPE["Pipeline and Workflow<br/>plan, execute, ensembles"]
        REG["Registry<br/>engines by (scale, kind)<br/>translator routing"]
        ENG["Engines<br/>solve one scale and kind"]
        TRN["Translators<br/>approximations and validate()"]
        IR["Scientific IR<br/>ScientificSystem, Quantity,<br/>provenance chain"]
        BK["SolverBackend<br/>eigensolvers and ODE<br/>(classical today)"]
    end

    subgraph QCB["Quantum-chemistry bridge (qc/)"]
        CACHE[("SQLite energy cache<br/>keyed by content")]
        WRK["PySCF worker<br/>in-process or inside WSL"]
    end

    subgraph STUDIES["Studies"]
        CAL["calibration.py<br/>fit the model to a surface"]
        TRF["transfer.py<br/>cross-prediction, rates,<br/>learning curves"]
        DIA["diatomics.py<br/>free X-H Morse curves"]
    end

    subgraph FRONT["Interfaces and outputs"]
        CLI["Command line<br/>run, calibrate, transfer, gui"]
        GUI["Mission control<br/>local web GUI"]
        RES["RunResult<br/>one JSON system per stage,<br/>with provenance"]
    end

    YAML --> EXP
    MOL --> EXP
    EXP --> PIPE
    PIPE --> REG
    REG --> ENG
    REG --> TRN
    ENG --- IR
    TRN --- IR
    ENG --> BK
    ENG -->|"qc_scan_2d"| CACHE
    CACHE <-->|"missing points"| WRK
    PIPE --> RES
    CLI --> PIPE
    GUI --> PIPE
    CLI --> CAL
    CLI --> TRF
    GUI --> TRF
    TRF --> CAL
    DIA --> CAL
    CAL -->|"reference surfaces, calibrated models"| PIPE
    TRF -->|"real and model chains"| PIPE
```

#### Scales, kinds and translators

Engines are keyed by `(scale, kind)` and translators accept particular kinds, so a chain is found kind-aware. Each arrow is a translator: a written list of
approximations plus a `validate()` that checks them against the actual numbers.

```mermaid
flowchart LR
    subgraph ES["electronic structure"]
        EVB1["evb_two_state<br/>1D model"]
        EVB2["evb_two_state_2d<br/>2D model"]
        QCS["qc_scan_2d<br/>real PySCF energies"]
    end
    subgraph QU["quantum"]
        DW["double_well_1d"]
        TAB["tabulated_1d"]
    end
    subgraph MO["molecular"]
        TRI["collinear_triatomic"]
    end
    subgraph RE["reaction"]
        NET["reaction.network<br/>mass-action ODE"]
    end
    subgraph BP["biophysical"]
        ENZ["enzyme_cycle"]
    end
    subgraph BI["biological"]
        PATH["metabolic_network"]
    end

    EVB1 -->|ElectronicToQuantum| TAB
    EVB2 -->|ElectronicToQuantum| TAB
    QCS -->|ElectronicToQuantum| TAB
    EVB2 -->|ElectronicToMolecular| TRI
    QCS -->|ElectronicToMolecular| TRI
    TAB -->|QuantumToReaction| NET
    DW -->|QuantumToReaction| NET
    TRI -->|MolecularToReaction| NET
    NET -->|ReactionToEnzyme| ENZ
    ENZ -->|BiophysicalToPathway| PATH
```

- **Two routes make different physics.** From a 2D electronic kind to `reaction` two chains tie for shortest, one through `quantum` (the proton is quantum, the heavy atoms
  are frozen) and one through `molecular` (relaxed heavy atoms, a classical proton). The registry raises `AmbiguousPathError` instead of picking one; name the
  intermediate scales in `propagation` to choose. Dividing one route's rate by the other's to get a "tunnelling factor" is wrong by four orders of magnitude at the defaults.
- `quantum.double_well_1d` is a starting kind only: nothing translates into it, so a chain can begin there.
- **Seams for other hardware and codes**: a `SolverBackend` (eigensolvers, ODE integration) where a GPU or quantum backend would plug in, and a `QCProgram` for external quantum-chemistry codes.
- **Uncertainty** is propagated by re-running the whole chain per draw. Parameters from a calibration are drawn *jointly* from their covariance, because treating them as independent inflates the rate uncertainty 30×.

### Low level

#### A run, step by step

`Pipeline.run` plans the chain, executes it as a DAG of solve and translate tasks, writes a provenance record for every product, and optionally repeats the whole chain per
uncertainty draw.

```mermaid
sequenceDiagram
    autonumber
    actor U as Caller (CLI, GUI or script)
    participant P as Pipeline
    participant R as Registry
    participant W as Workflow
    participant E as Engine
    participant T as Translator

    U->>P: run(root system, propagation, n_samples, seed)
    P->>R: plan with engine_for(scale, kind) and find_path(here, there, kind)
    R-->>P: ordered steps, or AmbiguousPathError / NoPathError
    P->>W: one task per step, each depending on the previous one

    loop every step in order
        alt a solve step
            W->>E: solve(system, backend)
            E-->>W: the system with its observables filled in
        else a translate step
            W->>T: validate(system)
            T-->>W: issues (an error stops the run, a warning is recorded)
            W->>T: translate(system)
            T-->>W: an unsolved system of the next scale, context parameters carried over
        end
        W->>W: append a ProvenanceRecord (fingerprints, approximations, warnings, seconds)
    end

    W-->>P: the trace, one system per step

    opt n_samples greater than 0
        P->>P: draw the root parameters (independent Gaussians or a joint covariance)
        P->>W: re-run the whole chain for each draw (a draw that fails validation is dropped)
        P->>P: attach the spread and a 68 percent band to every derived quantity
    end

    P-->>U: RunResult (trace, steps, ensemble information)
```

#### Core types

The Scientific IR, the contracts between components, and the two seams (numerical backend, quantum-chemistry program). Concrete engines and translators subclass
`Engine` and `Translator`; there is one per scale and kind in the diagram above.

```mermaid
classDiagram
    direction LR

    class ScientificSystem {
        +name
        +scale
        +kind
        +parameters
        +state
        +observables
        +structure
        +dynamics
        +provenance
        +param()
        +obs()
        +fingerprint()
        +evolve()
    }
    class Quantity {
        +value
        +unit
        +sigma
        +band
        +source
    }
    class ProvenanceRecord {
        +step
        +name
        +input_fingerprint
        +output_fingerprint
        +backend
        +approximations
        +warnings
        +seconds
    }
    class Engine {
        <<abstract>>
        +name
        +scale
        +kinds
        +approximations
        +solve()
    }
    class Translator {
        <<abstract>>
        +name
        +source
        +target
        +target_kind
        +source_kinds
        +approximations
        +validate()
        +translate()
    }
    class Registry {
        +register_engine()
        +register_translator()
        +engine_for()
        +find_path()
    }
    class Pipeline {
        +plan()
        +run()
    }
    class Workflow {
        +add()
        +run()
    }
    class RunResult {
        +trace
        +steps
        +ensemble
        +final
        +save()
    }
    class SolverBackend {
        <<abstract>>
        +lowest_eigenpairs()
        +symmetric_eigh()
        +integrate_ode()
    }
    class ClassicalBackend
    class QCProgram {
        <<abstract>>
        +compute()
    }
    class PySCFProgram

    ScientificSystem "1" *-- "many" Quantity
    ScientificSystem "1" *-- "many" ProvenanceRecord
    Registry o-- Engine
    Registry o-- Translator
    Pipeline --> Registry
    Pipeline --> SolverBackend
    Pipeline ..> Workflow : builds a DAG
    Pipeline ..> RunResult : returns
    Engine ..> SolverBackend : uses
    Engine ..> ScientificSystem : solves
    Translator ..> ScientificSystem : maps
    Engine ..> QCProgram : QCScanEngine only
    SolverBackend <|-- ClassicalBackend
    QCProgram <|-- PySCFProgram
```

#### The quantum-chemistry bridge

How a real surface is computed and cached. Every energy is stored by a content hash (geometry to 1e-8 Å, method, basis, charge, spin, program), so a rerun, and every draw of an
ensemble, reads from disk; only converged energies are kept.

```mermaid
sequenceDiagram
    autonumber
    participant S as QCScanEngine
    participant C as compute_cached
    participant DB as QCCache (SQLite)
    participant P as PySCFProgram
    participant W as pyscf_worker

    S->>S: build_geometry for every (x, R) of the grid (half of it when mirror symmetric)
    S->>C: jobs (atoms, charge, spin, theory, basis)
    C->>DB: get_many(content hash of every job)
    DB-->>C: the energies already known

    loop chunks of the missing jobs (40, or 6 for CCSD and CCSD(T))
        C->>P: compute(chunk)
        P->>W: jobs as JSON on stdin inside WSL, or a direct call where PySCF is installed
        W->>W: SCF, then MP2, CCSD or CCSD(T) if asked
        W-->>P: one result per job (energy, converged, HOMO-LUMO gap)
        P-->>C: results
        C->>DB: put_many(converged results only)
    end

    C-->>S: results, number computed, number cached
    S->>S: refuse non-converged points and verify a declared mirror symmetry
    S->>S: assemble E(x, R) in eV above the minimum, the rigid slice and the gap
```

#### The two-state model

The cheap model that stands in for real chemistry. The acceptor bond uses the donor's Morse parameters unless `acceptor_morse_*` are given (the separate-bond model).
Diagonalising the 2×2 Hamiltonian gives the ground-state surface the nuclei move on.

```mermaid
flowchart LR
    X["proton position x<br/>heavy-atom distance R"] --> RAB["r_A = R/2 + x<br/>r_B = R/2 - x"]
    RAB --> VA["V_A = Morse of r_A<br/>depth, width, r_eq"]
    RAB --> VB["V_B = Morse of r_B + offset<br/>acceptor curve: its own parameters,<br/>or the donor's"]
    X --> DL["coupling Δ(R)<br/>Δ0 exp(-β (R - R0))"]
    VA --> H["2x2 Hamiltonian<br/>V_A and V_B on the diagonal,<br/>Δ off the diagonal"]
    VB --> H
    DL --> H
    H --> E0["ground state E0<br/>the lower eigenvalue"]
    X --> VOO["V_OO(R)<br/>Morse in the heavy-atom distance"]
    E0 --> SUM["E(x, R) = E0 + V_OO(R)"]
    VOO --> SUM
    SUM --> O1["surface E(x, R)"]
    SUM --> O2["rigid slice at scan_distance"]
    H --> O3["electronic character and gap"]
```

#### Calibration and transfer

Fit the model to a real surface, then ask whether the fit says anything about another surface. Every comparison keeps the energy zero as its only freedom.

```mermaid
flowchart TB
    REF["Reference surface<br/>solved qc_scan_2d system:<br/>surface_x, surface_r, surface_energy"]
    ANC["diatomics.anchored_bounds<br/>(optional) hold the bonds<br/>at the free diatomics' values"]
    SET["FitSettings<br/>window, sigma, bounds, priors,<br/>reference distance, bond model"]
    FIT["calibrate_evb_2d<br/>weighted least squares,<br/>12 multi-starts"]
    CAL["Calibration<br/>parameters and covariance,<br/>pinned and poorly determined,<br/>wells and barrier per distance,<br/>leave-one-distance-out"]
    SYS["Calibration.system()<br/>a model system with a joint covariance:<br/>runs through the Pipeline in milliseconds"]
    RFO["Reference<br/>surface, calibration, fit window"]
    XP["cross_predict<br/>one calibration on another surface,<br/>best constant shift only"]
    RT["rate_comparison<br/>the real chain against the model chain<br/>at barrier-defined distances"]
    LC["learning_curve<br/>k distances plus a Gaussian prior<br/>from another calibration"]
    PT["parameter_table and baselines<br/>spread across references,<br/>what a constant guess scores"]
    OUT["Matrix, tables and curves<br/>in the CLI report and the GUI"]

    REF --> FIT
    SET --> FIT
    ANC -.-> SET
    FIT --> CAL
    CAL --> SYS
    REF --> RFO
    CAL --> RFO
    RFO --> XP
    RFO --> RT
    RFO --> LC
    RFO --> PT
    XP --> OUT
    RT --> OUT
    LC --> OUT
    PT --> OUT
```

#### Mission control

The GUI is a thin local web app over the same engines: the server chooses what to show, the page only draws.

```mermaid
flowchart LR
    B["Browser<br/>static HTML, CSS and JS<br/>no build step, no external requests"]
    subgraph SRV["gui/server.py"]
        H["Handler<br/>Host and Origin checks, JSON-only POST,<br/>strict Content-Security-Policy,<br/>serves static files by exact name"]
    end
    A["gui/app.py: App<br/>list and run experiments,<br/>transfer study, saved results"]
    J["JobManager<br/>background jobs the page polls"]
    PL["Pipeline<br/>experiments with edited inputs"]
    TS["transfer study<br/>references, matrix, rates,<br/>learning curves"]
    SV[("saved results<br/>keyed by file content hash")]
    QD[("QC energy cache")]
    SER["gui/serialize.py<br/>JSON, and which plots to draw<br/>at each scale"]

    B <-->|"JSON over HTTP, 127.0.0.1 only"| H
    H --> A
    A --> J
    A --> PL
    A --> TS
    TS --> SV
    TS --> QD
    PL --> SER
    TS --> SER
    SER --> H
```

The API the page uses: `GET /api/status`, `/api/experiments`, `/api/experiment`, `/api/job`, `/api/references`, `/api/transfer` and `POST /api/run`, `/api/transfer/run`,
`/api/transfer/rates`, `/api/transfer/learning`. Experiment paths are resolved so they cannot leave `experiments/`.

## What it found

The details, with every number and the predictions that failed, are in [docs/findings.md](docs/findings.md).

- **An isotope effect survives from the molecule to the pathway flux only if three conditions hold.** The chemical step has an intrinsic H/D effect of ~10; the flux effect is 10.0 when chemistry limits within the enzyme,
  the enzyme is saturated, and it controls the pathway, and falls to 1.00, 1.11 or 1.00 when any one of them is broken.
- **A calibrated model can stand in for real chemistry, for one molecule.** Fitted to the Hartree–Fock/6-31G* surface of the Zundel cation it reproduces it to 0.01 eV where the nuclei sample, and its rates stay within 0.72–1.21× of
  the real chain across nine orders of magnitude, at ~35 ms per chain instead of minutes of energies.
- **Fitted parameters do not transfer.** Each surface fits to 0.01–0.06 eV; the same parameters on another method's surface of the same molecule cost 0.16–0.25 eV, and on another molecule a median of 0.34–0.39 eV, ten times the own-fit
  error and worse than predicting one constant for 89 of 420 pairs.
- **That is not an artifact of letting the bonds float.** The bond lengths and widths *are* physical (within a few percent of the free diatomics). Pinning them there makes the fits much worse (median 0.031 → 0.119 eV) and
  transfer no better: the coupling, the heavy-atom well and the offset, which decide the barrier, are what depends on the molecule and the method.
- **A pair whose two bonds differ needs a curve of its own for each.** The chloride–hydrogen fluoride ion cannot be fitted with one Morse curve (0.41 eV); with `bonds: separate` it fits to 0.03–0.04 eV at HF, B3LYP and MP2,
  with equilibrium lengths near those of free HCl and HF.
- **The methods themselves disagree, in a known direction.** A CCSD(T) benchmark of the bifluoride ion puts HF's barrier too high and B3LYP's too low, and the study's own surfaces miss the CCSD(T)/aug-cc-pVTZ barrier at 2.74 Å by
  +0.19 (HF), −0.19 (B3LYP) and −0.14 eV (MP2), about a thousandfold in a rate at 300 K.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/benchmark-barrier-dark.png">
  <img alt="The bifluoride proton-transfer barrier against F···F distance for HF, B3LYP, MP2 and CCSD(T) in 6-31+G* and aug-cc-pVTZ" src="docs/images/benchmark-barrier-light.png" width="900">
</picture>

## Scientific status

Read [docs/limits.md](docs/limits.md); in brief:

- The cheap model's parameters are **effective, not physical**: fitted couplings are 3–20 eV, and the fitted energy offset is 2–3× the real well gap at Hartree–Fock and larger still at B3LYP and MP2. A good fit shows flexibility, not that the model is right.
- The scans are **rigid, collinear and gas-phase**, with small basis sets. Against CCSD(T)/aug-cc-pVTZ for one ion, the study's barriers are off by 0.14–0.19 eV at 2.74 Å, which is far too much for absolute rates.
- The study has **seven molecules, not twenty-one independent samples** (the methods of one molecule are correlated), and its ordering by "chemical distance" rests on two parent–derivative pairs.
- That EVB parameters are system-specific is already known in the literature; the contribution here is measuring it carefully, with the failed predictions kept in.
- The scales above chemistry (enzyme, pathway) run on free inputs and show *how* lower-scale chemistry is expressed or hidden, not what a real system does.

## Validation

429 tests: 365 run without PySCF (1.5–5 minutes), and 64 need a real quantum-chemistry program. They use closed-form results and independent calculations instead of the code agreeing with itself:

- The two-state model equals the engine's diagonalisation to 1e-10; the closed-form calibration recovers a noise-free surface exactly and reports honest uncertainty on a noisy one.
- The PySCF bridge is checked against physics: the variational principle and the Hartree–Fock limit, invariance under rigid motion, CCSD being exact for two electrons (so the triples correction vanishes),
  and the free-diatomic zero-point energy against a numerically solved Schrödinger equation.
- The tests can fail: well over a hundred one-line mutants of the calibration, transfer, separate-bond, diatomics, engine and command-line code were run; every survivor was either a real test gap, now closed, or
  documented as an equivalent mutant.
- Bugs found in this project's own code are listed in [docs/validation.md](docs/validation.md), including a thread-safety bug in the ODE solver that two browser tabs exposed.

## Repository layout

```
src/substrate/
  ir.py  base.py  pipeline.py  workflow.py      the Scientific IR, engine/translator contracts, routing, provenance and ensembles
  engines/                                      electronic (model, and real quantum chemistry), molecular, quantum, reaction, biophysical, biological
  translators/                                  the documented, validated hops between scales
  qc/                                           PySCF bridge: jobs, SQLite energy cache, a standalone worker that runs inside WSL
  calibration.py  transfer.py  diatomics.py     fit the model to a surface; cross-prediction and learning curves; free X–H Morse curves
  molecules.py                                  templates: Zundel, ammonium dimer, bifluoride, water–ammonia, methanol–water, ...
  gui/                                          the local web GUI: standard-library server, plain HTML/JS, no build step, no external requests
experiments/                                    runnable experiment files; references/ (24 real surfaces) and benchmarks/ (8 bifluoride surfaces)
examples/                                       the scripts in the table above
tests/                                          429 tests
docs/                                           architecture, science, findings, validation, limits, extending, images
```

## Documentation

| | |
|---|---|
| [docs/findings.md](docs/findings.md) | What the studies measured, with numbers and the failed predictions |
| [docs/limits.md](docs/limits.md) | What the numbers can and cannot support |
| [docs/science.md](docs/science.md) | The implemented science, scale by scale |
| [docs/architecture.md](docs/architecture.md) | The IR, routing, backends, the QC bridge, the GUI |
| [docs/validation.md](docs/validation.md) | How each part is tested, and the bugs the tests found |
| [docs/extending.md](docs/extending.md) | Adding the next scale or model |

## Development

```bash
pip install -e ".[dev]"            # add ".[plots]" for the benchmark figure
pytest                             # everything; tests marked `qc` need PySCF and skip without it
pytest -m "not qc"                 # the 365 that do not
```

## Credits and license

Quantum chemistry is computed with [PySCF](https://pyscf.org); diatomic spectroscopic constants are from the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/).
Numerics use NumPy and SciPy.

No license has been chosen yet, so the default copyright applies (all rights reserved). Open an issue if you would like to use or contribute to the project.
