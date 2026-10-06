# substrate

Infrastructure for propagating scientific models across scales, through one common representation,
with every hop documented and validated.

```
                              ┌─▶ quantum ──────▶┐
electronic structure ─────────┤                  ├─▶ reaction ─▶ biophysical ─▶ biological
                              └─▶ molecular ────▶┘
```

Nothing crosses a scale boundary except through a **Translator**: a written list of approximations plus
a `validate()` that checks them against the actual numbers. If the physics doesn't support the hop, the
pipeline stops with a reason instead of inventing a result.

## What exists today

Every declared scale now has an engine. One proton transfer can be followed from an electronic Hamiltonian to the flux of a
metabolic pathway, nine stages and five scales, with provenance, uncertainty and validity checks at every hop.

| Scale | Status |
|---|---|
| `electronic_structure` | **Two engines.** (1) A model: two-state valence-bond (EVB) Hamiltonians diagonalised at every geometry, a 1D scan or a 2D surface; its parameters are inputs, **not ab initio**. (2) **Real quantum chemistry** (`electronic.qc_scan_2d`): a 2D surface from single-point energies computed by PySCF (Hartree–Fock, MP2 or DFT) for any molecule with a donor–proton–acceptor axis. Both feed the rest of the chain identically. |
| `molecular` | **Implemented.** Atoms with masses on the 2D surface: relaxed profile, stationary points, harmonic normal modes, thermochemistry, classical transition-state-theory rates. |
| `quantum` | **Implemented.** Nuclear quantum dynamics of one particle on a 1D double well (analytic or tabulated from the electronic scale): levels, tunnel splitting, barrier transmission. |
| `reaction` | **Implemented.** Mass-action kinetics (ODE) for any reaction network. |
| `biophysical` | **Implemented.** An enzyme cycle `E + S ⇌ ES ⇌ EP ⇌ E + P` whose chemical step comes from the scales below: k_cat, K_M (forward and reverse), k_cat/K_M, binding free energy, populations, flux control. |
| `biological` | **Implemented.** A metabolic network at steady state (rate-law library, stability check, time course) with metabolic control analysis. The standard instance is `external substrate → S → enzyme → P → drain`, with the enzyme's kinetics supplied by the biophysical scale. |

**Mission control** (`python -m substrate gui`, then open http://127.0.0.1:8765/). A local web GUI over the same engines, with no extra
dependencies. *Run*: pick any experiment, edit any input (or give it a σ to turn on an ensemble) and the whole chain re-runs; every scale shows as a
card in the pipeline strip with its time, its warnings and what it assumed, the stage you select shows its plots (the surface E(x, R), the potential
with its energy levels, species over time, control coefficients) and every observable beside its change from the previous run, and a hop the physics
refuses (the barrier fell below the zero-point energy) stops the strip at that stage with the reason and keeps everything that completed. *Transfer
study*: the eighteen-reference study of the next section as a clickable matrix (click a cell to see the target's real energies against the source's
parameters and against its own fit, at any distance), the own-fit and parameter tables, the downstream rates, and the few-shot learning curves; results
are saved and reload instantly. Every plot has a hover readout, a table view and a light/dark theme. It listens on 127.0.0.1 only and runs only the
experiment files in `experiments/`.

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

The headline results:

- **An isotope effect survives from the molecule to the pathway flux only if three separate conditions hold**
  (`python examples/pathway_isotope_effect.py`). The molecular scale computes an intrinsic H/D effect of ~10 for the chemical step.
  The effect on the pathway flux is 10.0 when (1) chemistry limits within the enzyme, (2) the enzyme is saturated, and (3) the enzyme
  controls the pathway. Break any one and it vanishes: slow product release gives 1.00 (release hides the chemistry), an
  unsaturated enzyme gives 1.11 (the flux follows k_cat/K_M, which for a sticky substrate is just the binding rate), and a
  transport-limited pathway gives 1.00 (flux-control coefficient of the enzyme: 0.000).
- **Control is not local.** Slowing transport shifts control away from the enzyme, and a first-order prediction
  `KIE_flux ≈ KIE_kcat ^ C_enzyme` works where one step clearly controls the pathway for both isotopes and fails in between: at
  transport = 0.3 s⁻¹ it predicts 1.02 but the true effect is 3.81, because deuteration makes the enzyme 10× slower and hands
  control back to it. A local estimate made at H cannot see that.
- **The two chemistry routes can disagree by 10⁴ and still agree on k_cat.** With a slow release only the chemistry's
  *equilibrium* matters, and both routes agree on it: k_cat differs by 14% between routes whose chemical rate constants differ by a
  factor of ~16,000 (`python examples/enzyme_isotope_effects.py` explores the commitment regimes).
- **Real quantum chemistry on a real ion** (`python examples/zundel_real_chemistry.py`; Hartree–Fock/6-31G* energies from PySCF for the
  Zundel cation H₅O₂⁺, rigid water-like groups). The barrier to moving the proton grows from 0.05 to 0.93 eV as the O···O distance goes from
  2.5 to 3.0 Å, confirming the model engine's central assumption. It also shows that the model's *numbers* were illustrative: its barriers
  at the same distances are 2–12× too large (0.57 vs 0.05 eV at 2.5 Å, 1.7 vs 0.93 eV at 3.0 Å). On the relaxed surface the oxygens approach and the proton is
  shared, so there is no proton-transfer reaction and the molecular route refuses, saying why (zero-point energy exceeds the tiny
  barrier). With the O···O distance held fixed, as an enzyme active site would hold it, the quantum route gives k_H falling from 3×10¹³ s⁻¹
  at 2.6 Å to 8×10⁴ s⁻¹ at 3.0 Å, with an H/D isotope effect growing from 9 to ~10⁴ as tunnelling takes over. The same chain
  then runs on to the pathway flux (nine stages from Hartree–Fock energies).
- **Calibrating the cheap model against the real surface** (`python examples/calibrate_against_real_chemistry.py`, or
  `python -m substrate calibrate experiments/zundel_hf.yaml`). The model engines' default parameters were off from the real chain by
  4–6 orders of magnitude in rate. Fitted to the Hartree–Fock surface, the 2D model reproduces it to 0.01 eV RMSE in the region the nuclei
  sample, gets the number of wells right at every distance, and predicts a distance it never saw to 0.011 eV in barrier height. Downstream, its
  rates are within a factor of 0.72–1.21 of the real chain across **nine orders of magnitude** (2.6–3.0 Å) and its isotope effects within ~20%.
  A whole chain to a rate costs ~35 ms against ~0.3–0.5 s per real energy, so a 100-draw ensemble over a new heavy-atom distance is seconds instead of ~10 minutes. **The fit's own uncertainty (6% in the rate) is smaller than the model's
  actual error (up to 28%)**: it measures how far the parameters can move, not what a fixed functional form cannot represent. The fitted
  parameters are strongly correlated, so ensembles draw them *jointly*; treating them as independent inflates the rate uncertainty 30×.
- **Do the fitted parameters transfer? Not across molecules, and not across levels of theory unless the theories are alike**
  (`python examples/transfer_across_molecules_and_methods.py`, or `python -m substrate transfer experiments/references/*.yaml`). Nine real
  symmetric surfaces: the Zundel cation H₅O₂⁺ (O–H–O) at HF, B3LYP and MP2 with 6-31G* and at HF/cc-pVDZ; the ammonium dimer N₂H₇⁺ (N–H–N) at HF and
  B3LYP; and the bifluoride ion FHF⁻ (F–H–F) at HF, B3LYP and MP2 with 6-31+G*. Each is fitted on its own to 0.009–0.019 eV RMSE with the right number
  of wells at every distance, so the model form is not the obstacle. Then every calibration is applied, parameters unchanged, to every reference
  (72 pairs; the only freedom is the arbitrary energy zero). Against the error of a reference's own fit: changing only the **basis set** costs 3–4×
  (0.037–0.039 eV); **B3LYP ↔ MP2** on the same molecule costs 3× (0.028–0.031 eV); **Hartree–Fock ↔ B3LYP or MP2** on the same molecule costs
  10–23× (0.155–0.209 eV, about half the spread of the surface itself); and **another molecule** costs 8–88× (0.10–0.91 eV, median 0.34), which is *worse
  than predicting one constant energy* for 22 of the 72 pairs, because the surfaces are 0.33–0.40 eV wide. The same sorting shows in the real chain's rates
  at the distance where each target's barrier is 0.4 eV: a calibration's own rate is within 0.60–1.10× of the real chain's, a basis change within 0.36–1.4×,
  B3LYP ↔ MP2 within 0.14–5×, but HF ↔ B3LYP/MP2 off by a median of ~550× (4×10⁻⁵ to 3×10²) and another molecule by a median of ~50× (5×10⁻¹¹ to 2×10³).
  (The method really does move the surface: at each molecule's scan distance (2.8, 2.9 and 2.7 Å) the barrier is 0.49, 0.50 and 0.53 eV at Hartree–Fock
  and 0.18, 0.19 and 0.19 eV at B3LYP, for the Zundel ion, the ammonium dimer and bifluoride.)
  The parameters that carry a physical meaning *do* look alike (the X–H Morse width within 1.00–1.07× across the methods of one molecule and 1.06× across the
  four symmetric Hartree–Fock references; the X–H length 0.95–1.07 Å over all nine); the ones that absorb what the model form cannot represent do not: the coupling
  differs by up to 2.2× across the methods of one molecule, and the O···O well depth by up to 5.9× within one molecule and 10× across the four symmetric Hartree–Fock
  references (25× with the three asymmetric molecules' HF results added), and those decide the barrier. What does carry over is *information as a prior*: with the same molecule's Hartree–Fock result as the prior (the other basis, for the two Zundel HF targets), a
  single computed heavy-atom distance predicts the held-out distances to 0.022–0.098 eV across the three prior widths tried (from scratch one distance cannot be
  fitted at all, two give 0.04–0.4 eV, three 0.016–0.044), but another molecule's parameters are no better at one distance than using them unchanged (0.18–0.40
  eV), and at two distances they help about as much as the same-molecule prior (0.025–0.135 eV both ways), which says they act as a regulariser, not as knowledge.
- **An asymmetric surface, the harder test of the model form** (the water–ammonia cation [H₂O···H···NH₃]⁺, O–H–N, at HF, B3LYP and MP2 with 6-31G*; every proton position is computed, 266–275 energies each). Ammonia holds a proton far more tightly than water, so the proton sits on the nitrogen: at HF there is one well up to O···N = 2.7 Å and a metastable O-side well appears from 2.8 Å, 0.99 eV higher there and 1.64 eV higher at 3.4 Å. The model form still fits it, but not as well, and only with a wider fit window and its energy offset: with the window raised to 2.5 eV (declared in the experiment files, `calibration: {window_ev: 2.5}`, because the barrier top lies 1.0–2.7 eV above the minimum (HF; 1.1–2.1 at B3LYP, 1.2–2.3 at MP2) and the default 1.5 eV window never sees it, while 2.5 eV reaches it everywhere except HF at 3.4 Å; at 1.5 eV the 3.4 Å barrier is 0.21 eV too low, at 2.5 eV 0.10) the RMSE is 0.040–0.043 eV against 0.009–0.019 for the symmetric surfaces, the worst error 0.17–0.18 eV, the well count right at all 11 distances, and the barrier (measured from the oxygen-side well) within 0.011–0.039 eV on average; leave-one-distance-out predicts an unseen distance to 0.036–0.038 eV (HF, MP2). **The offset does the work**: forced to zero, the fit is ten times worse (0.45 and 0.42 eV against 0.043 and 0.040 at HF and MP2) and loses a well at 2–3 of 11 distances. It is an *effective* parameter, not a proton-affinity difference: −2.11 ± 0.10 eV at HF, −4.65 ± 0.54 at B3LYP, −3.47 ± 0.27 at MP2 (and the coupling's decay sits on its lower bound, 0, for B3LYP and MP2). The transfer picture is the symmetric one made worse. Between methods on this molecule, HF ↔ B3LYP/MP2 costs 4–6× the own-fit error (0.17–0.23 eV) and B3LYP ↔ MP2 2× (0.074 eV). Between a symmetric and an asymmetric surface (54 pairs, in both directions) the error is 0.36–0.76 eV (median 0.49), 10–69× the own-fit error, and above the constant-guess baseline for 27 of the 54, with the wrong number of wells at a median of 5 of 11 distances; the rates miss by a median of ~2×10⁸ (at 0.15 eV), 10⁸ (0.4 eV) and 4×10⁶ (0.8 eV). A calibration's own rate is within 0.67–0.96× of the real chain's for this ion at 0.15 and 0.4 eV (2.1× at 0.8 eV, HF only), but HF ↔ B3LYP/MP2 is off by a median of 2,000–3,000× at those two barriers. Few-shot: from one distance a fit *proceeds* (its 2.5 eV window holds enough points) and is wildly wrong on the held-out distances (1.1, 79 and 77 eV), two give 0.09–0.25 eV and three 0.051–0.11; with the same ion's HF result as the prior one distance gives 0.14–0.36 eV, two 0.10–0.14 and three 0.044–0.049, against the 0.022–0.098 that one distance gave for the symmetric molecules, and the offset is part of that prior. (At the scan distance 3.1 Å the donor-side barrier is 0.45 eV at HF, 0.16 at MP2 and 0.088 at B3LYP.)
- **A second asymmetric molecule, a milder one, and what it shows about distance** (the methanol–water cation [H₂O···H···HOCH₃]⁺, O–H–O, at HF, B3LYP and MP2 with 6-31G*, 276 energies each). Methanol holds a proton only ~0.65 eV more tightly than water (water–ammonia: ~1.7 eV), so it sits between the symmetric Zundel ion (the same O–H–O chemistry) and water–ammonia. At HF the proton is on the methanol, one well up to 2.56 Å and two from 2.64 Å, the water-side well 0.45–0.66 eV higher (0.40–0.55 at B3LYP, where the second well appears from 2.80 Å, and 0.36–0.55 at MP2, from 2.72 Å). Its fit window was set by a rule, not by looking at the fit: the smallest of 1.5, 2.0, 2.5 or 3.0 eV that reaches the barrier top at every double-well distance (HF tops reach 2.16 eV, so 2.5). The model form fits it, but **not better than the stronger asymmetry**: RMSE 0.051, 0.059 and 0.053 eV (HF, B3LYP, MP2) against 0.040–0.043 for water–ammonia and 0.009–0.019 for the symmetric surfaces, worst error 0.23–0.27 eV, the well count right at every distance but one (MP2), barriers within 0.009–0.013 eV on average, and the coupling's decay pinned at its lower bound (0) in all three; leave-one-distance-out (HF) 0.051 eV. The offset matters less here: forced to zero the HF fit is 4.7× worse (0.24 against 0.051 eV) and loses a well at 2 of 11 distances, where water–ammonia was 10× worse. It is again an effective parameter: −1.66 ± 0.14, −3.57 ± 0.94 and −2.56 ± 0.43 eV at HF, B3LYP and MP2, and its ratio to water–ammonia's at the same method (1.3–1.4) is nowhere near the ratio of the proton-affinity gaps (about 2.6). **Transfer follows the chemical distance.** Same method on both sides (so only the chemistry differs), the cross-prediction error is smallest between the symmetric parent and its one-methyl derivative, larger from the derivative to the strongly asymmetric ion, and largest from the parent to the strongly asymmetric ion, at every one of the three methods (eV, source → target):

  | | Zundel → methanol–water | methanol–water → Zundel | methanol–water → water–ammonia | water–ammonia → methanol–water | Zundel → water–ammonia | water–ammonia → Zundel |
  |---|---|---|---|---|---|---|
  | HF | 0.247 | 0.218 | 0.312 | 0.302 | 0.491 | 0.421 |
  | B3LYP | 0.208 | 0.175 | 0.269 | 0.264 | 0.417 | 0.360 |
  | MP2 | 0.205 | 0.176 | 0.289 | 0.278 | 0.451 | 0.389 |

  Adding one methyl to the symmetric ion costs about as much (0.18–0.25 eV, 4–23× the own-fit error) as switching from Hartree–Fock to B3LYP or MP2 on the symmetric ion (0.16–0.21 eV); the two asymmetric molecules together (18 pairs) are 0.26–0.44 eV apart (median 0.28), none worse than a constant guess, where 108 symmetric ↔ asymmetric pairs have a median of 0.42 eV (0.18–0.76) and 34 of them are worse than a constant. The rates are much noisier and do not show the ordering cleanly (median miss, symmetric ↔ asymmetric: 4×10⁵, 2×10⁴ and 4×10⁴ at the 0.15, 0.4 and 0.8 eV barriers; between the two asymmetric molecules 1×10⁴, 6×10⁴ and 2×10³). Few-shot learning on methanol–water is the best of the asymmetric cases: with its own HF result as the prior, one distance gives 0.071–0.073 eV at the tightest width (floor 0.052–0.057 eV) and 0.17–0.31 eV at the looser ones, and two distances reach the floor (0.051–0.060); from scratch one distance gives 0.8–10 eV (the fit proceeds and is wrong), two 0.28–0.44 and three 0.047–0.061.
- **A third asymmetric molecule, the nitrogen counterpart: the ordering by chemical distance generalises** (the ammonia–methylamine cation [H₃N···H···H₂NCH₃]⁺, N–H–N, at HF, B3LYP and MP2 with 6-31G*, 276 energies each, 12 atoms). It is the one-methyl derivative of the symmetric ammonium dimer, as methanol–water is of the Zundel ion, and a milder asymmetry again: methylamine holds a proton ~0.47 eV more tightly than ammonia, and the two wells differ by 0.25–0.42 eV at HF (0.26–0.37 at B3LYP, 0.22–0.37 at MP2), the second well appearing from 2.68, 2.86 and 2.77 Å. The fit window comes from the same rule (HF barrier tops reach 2.05 eV, so 2.5). These are the best fits of the asymmetric surfaces: RMSE 0.039, 0.050 and 0.043 eV (HF, B3LYP, MP2), worst error 0.17–0.24 eV, every well count right, barriers within 0.014–0.023 eV on average, the coupling's decay on its lower bound in all three, leave-one-distance-out (HF) 0.039 eV with a barrier error of 0.021. **The fitted offset is the smallest again, and ordered as the proton-affinity gaps at every method, without being proportional to them**: ammonia–methylamine −0.83 ± 0.04, −1.46 ± 0.19 and −1.10 ± 0.10 eV (HF, B3LYP, MP2), methanol–water −1.66, −3.57, −2.56, water–ammonia −2.11, −4.65, −3.47 (the ratios to the smallest are 2.0–2.4 and 2.5–3.2, against gap ratios of 1.4 and 3.6). Forced to zero, the HF fit is 3.9× worse (0.155 against 0.039 eV) and loses a well at 1 of 11 distances (methanol–water 4.7×, water–ammonia 10×). **Transfer follows the chemical distance in nitrogen too.** In all five parent / derivative / strongly-asymmetric triples the study contains (the Zundel family at three methods, the ammonium-dimer family at two; there is no MP2 ammonium dimer) and in both directions, the error is smallest between the symmetric parent and its one-methyl derivative, larger from the derivative to water–ammonia, and largest from the parent to water–ammonia; for the ammonium dimer (parent → derivative, derivative → water–ammonia, parent → water–ammonia) 0.159, 0.355 and 0.492 eV at HF and 0.140, 0.311 and 0.420 at B3LYP. The parent ↔ derivative errors are smaller for the milder asymmetry (nitrogen 0.128–0.159 eV, oxygen 0.175–0.247), and per eV of proton-affinity gap all ten of them are 0.27–0.38 eV of error: an observation about two derivatives, not a law. A change of heavy atom between the two derivatives costs 0.32–0.42 eV. Over the 18 references, the pairs of *different asymmetric molecules* (54) are 0.24–0.57 eV apart (median 0.33, 4–12× the own-fit error) and none is worse than a constant guess, while the 162 symmetric ↔ asymmetric pairs have a median of 0.42 eV (0.13–0.92) and 52 are worse than a constant; of all 306 pairs 74 are. A calibration's own rate is within 0.6–1.0× of the real chain's at the 0.4 eV barrier for all 18 references. **Few-shot learning, with four kinds of prior** (tightest width, one distance; floors 0.045–0.058 eV): the same molecule's HF result 0.063–0.079 eV, its symmetric parent's 0.087–0.14, a different heavy atom's (Zundel) 0.32–0.38, nothing (the fit proceeds and is wrong) 0.69–39 eV; at two distances 0.067–0.072, 0.080–0.104, 0.13–0.14 and 0.13–0.17; at three 0.054–0.061, 0.050–0.062, 0.051–0.063 and 0.074–0.075. At one and two distances the prior's usefulness falls in the order of chemical distance; by three distances the three priors are within 0.012 eV of each other.
- Molecular scale: the O···O distance **relaxes from 2.64 Å in the wells to 2.39 Å at the transition state**. The relaxed barrier is
  0.53 eV, against 1.07 eV if the heavy atoms were frozen at the well distance (0.27 eV if frozen at the TS distance: which
  distance you freeze matters enormously). Frequencies are physical (O–H stretch 3525 cm⁻¹, O···O stretch 404 cm⁻¹, imaginary mode
  2588i cm⁻¹ at the TS).
- Electronic structure: shrinking O···O from 2.7 to 2.35 Å drops the barrier from 1.06 to 0.27 eV while k_H rises from 1e7 to
  3e13 s⁻¹ (`examples/donor_acceptor_sweep.py`). At 2.30 Å the barrier is below the zero-point energy and no rate is produced.

Not built: a 2D nuclear quantum treatment, relaxed (rather than rigid-group) scans for the quantum-chemistry engine, a model form that
transfers (the fitted parameters were tested on eighteen references, nine of them symmetric and nine of three asymmetric molecules, and do not), regulation / gene expression /
stochasticity at the biological scale, reverse propagation, a QPU/GPU backend, the data fabric, the registry/artifact store beyond
JSON-on-disk, and any UI beyond the CLI report.

In progress, not yet written up here: two more asymmetric anions, the fluoride–methanol ion [F···H···OCH₃]⁻ (F–H–O, mild; its templates, experiment files and tests are in the repository, but the full study with it has not been run and this README's numbers do not include it) and the chloride–hydrogen fluoride ion [Cl···H···F]⁻ (Cl–H–F, strong; the template and a first Hartree–Fock experiment file exist, with the scans not yet validated).

## Quickstart

```bash
pip install -e ".[dev]"
python -m substrate run experiments/proton_transfer.yaml --out runs/pt001      # quantum scale only
python -m substrate run experiments/proton_transfer_electronic.yaml            # electronic -> quantum -> reaction
python -m substrate run experiments/proton_transfer_molecular.yaml             # electronic -> molecular -> reaction
python -m substrate run experiments/proton_transfer_enzyme.yaml                # ... -> reaction -> biophysical
python -m substrate run experiments/proton_transfer_pathway.yaml               # ... -> biophysical -> biological
python -m substrate run experiments/zundel_hf.yaml                             # REAL quantum chemistry (needs PySCF, see below)
python examples/zundel_real_chemistry.py
python -m substrate calibrate experiments/zundel_hf.yaml --out experiments/zundel_calibrated.yaml   # fit the model to the real surface
python -m substrate run experiments/zundel_calibrated.yaml                     # ... and run it in seconds, with correlated uncertainty
python examples/calibrate_against_real_chemistry.py
python -m substrate gui                                                        # mission control in the browser (--port, --no-browser)
python -m substrate transfer experiments/references/zundel_hf.yaml experiments/references/fhf_hf.yaml   # does a fit carry to another molecule?
python examples/transfer_across_molecules_and_methods.py --only learning      # just the few-shot section (needs the energies cached)
python examples/transfer_across_molecules_and_methods.py                       # eighteen references, all comparisons (computes ~3,700 energies the first time, ~3 hours here)
python examples/pathway_isotope_effect.py
python examples/enzyme_isotope_effects.py
python examples/molecular_vs_quantum.py
python examples/donor_acceptor_sweep.py
pytest                                                                          # 1.5-4 min without PySCF, 2-6 min with a warm energy cache (depends on load); tests marked `qc` need PySCF and skip without it
```

**Real quantum chemistry needs PySCF**, which has no Windows build. On Windows, `scripts/setup_qc_env.sh` creates an isolated
environment inside WSL (no `sudo`, nothing system-wide, everything in `~/.substrate-qc`; remove it with `rm -rf ~/.substrate-qc`):

```bash
wsl bash /mnt/c/<path to this project>/scripts/setup_qc_env.sh
```

On Linux or macOS `pip install pyscf` is enough. The engine finds PySCF itself (in-process if importable, otherwise through WSL);
`SUBSTRATE_QC_MODE`, `SUBSTRATE_QC_WSL_DISTRO` and `SUBSTRATE_QC_WSL_PYTHON` override discovery. Computed energies are cached on disk by
content (`~/.cache/substrate`, or `SUBSTRATE_CACHE_DIR`), so the first run of a scan costs minutes and every later run, including each
draw of an uncertainty ensemble, is instant.

```yaml
experiment:
  phenomenon: pathway_flux_from_proton_transfer
  system:
    scale: electronic_structure
    kind: electronic.evb_two_state
    parameters:
      coupling: {value: 0.6, unit: eV}
      context.k_release:    {value: 1.0e4, unit: 1/s}                     # biophysical input, carried across every scale
      context.enzyme_total: {value: 1.0e-6, unit: M, sigma: 2.0e-7}       # biological input; sigma => Monte-Carlo through the chain
      ...
  propagation: [biological]                                                # naming only the destination is enough when unambiguous
```

## Architecture

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
  geometries as `QCJob`s (atoms, charge, spin, theory, basis); a `QCProgram` returns `QCResult`s. `PySCFProgram` runs PySCF
  in-process if it is installed, or through a standalone worker (`pyscf_worker.py`, which imports nothing from substrate) in WSL, over a
  JSON protocol, in chunks so a crash loses at most one chunk. Energies are cached in SQLite by a content hash (geometry to 1e-8 Å,
  method, basis, charge, spin, program). A missing program raises `QCError`, deliberately *not* a `ValidationError`: an ensemble drops
  invalid draws, but it must stop for an environment failure. Non-convergence is reported through the result and becomes a
  `ValidationError`.
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
  rates against the same chain on each calibration, and few-shot learning curves with a Gaussian prior (see below).
- **Experiment files** can carry a non-numeric `structure:` (a molecule, a method, a mechanism) alongside numeric parameters.
- **`calibration.py`: fitting the model engines to a reference surface** (see below), and **joint parameter draws** in the ensemble: a
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

## The implemented science

**Electronic structure.** For a proton at displacement `x` from the midpoint of two heavy atoms a distance `R` apart
(`r_A = R/2 + x`, `r_B = R/2 - x`):

```
H = [[ Morse(r_A),   Δ(R)                ]      E₀ = lower eigenvalue  →  the surface the nuclei move on
     [ Δ(R),         Morse(r_B) + offset ]]
```

`electronic.evb_two_state` fixes `R` (1D). `electronic.evb_two_state_2d` makes `R` a coordinate with
`Δ(R) = Δ₀ exp(-β(R-R₀))` and a Morse O···O interaction; it also reports a rigid slice at `scan_distance` cut from the *same*
surface. (A soft harmonic O···O term was tried first and fails: the coupling grows exponentially as `R` shrinks and the surface
collapses into one symmetric short-distance well.) The Morse defaults give a realistic O–H stretch (~3500 cm⁻¹).
Approximations: two valence-bond states only, shared Morse parameters, collinear geometry, no environment, ground state only.

**Real quantum chemistry** (`electronic.qc_scan_2d`). The same observables from real energies. A molecule with its donor, proton and
acceptor on a z axis is described in the experiment's `structure`; every other atom belongs to a rigid group that travels with the donor or
the acceptor. For each proton displacement `x` and heavy-atom distance `R` on a grid, the engine builds the geometry, asks the program for a
single-point energy (`hf`, `mp2` or `dft:<functional>`, any basis, open-shell handled by the unrestricted variants), and assembles
`E(x, R)` in eV above the minimum, plus the rigid slice at `scan_distance` (so the quantum route stays available) and the HOMO–LUMO gap. If the molecule is
symmetric under exchanging donor and acceptor, `mirror_symmetric` computes only `x ≥ 0`, and the claim is **verified on a pair of points, not
assumed**. `substrate.molecules.zundel_cation()` is the worked example. Approximations: the level of theory and basis set; a rigid scan (group
internals do not relax as the proton or the heavy atoms move); a collinear D–H–A axis; gas phase; and a HOMO–LUMO gap that is only a proxy for
the excitation gap (Hartree–Fock overestimates it).

**Calibration** (`substrate.calibration`, `python -m substrate calibrate`). Fits the 2D valence-bond model to any solved system that exposes
`surface_x`, `surface_r`, `surface_energy` (typically the quantum-chemistry engine's), by weighted least squares with multi-start. Modelling
choices that change the answer are explicit settings and recorded in the result: only points up to 1.5 eV above the surface minimum are fitted
(the nuclear problem samples the surface up to the barrier top plus a few zero-point energies; a far wall at tens of eV would otherwise
dominate) with a flat 10 meV tolerance (rates depend exponentially on energies; a relative tolerance let the barrier region drift by 0.07 eV
and cost a 2.5× rate error). The coupling is defined at the middle of the distance range, which decorrelates it from its decay. Bounds are
generous because a tight one silently becomes the answer: an earlier cap of 3 eV on the coupling sat on the data's optimum of 4.1 eV. The
result reports, besides the parameters, the fit quality by energy window; per heavy-atom distance the number of wells, the barrier and the
well position for reference and model; **which parameters sit on a bound and which are poorly determined**; and leave-one-distance-out
validation (a bounded number of held-out distances, each refit warm-started). The analytic double well (`quantum.double_well_1d`) calibrates
to a 1D slice the same way, and a 1D model is cut from the calibrated 2D one at any distance. Uncertainty is a Laplace approximation scaled
by the misfit (`max(χ²_red, 1)·(JᵀJ)⁻¹`) and is attached as marginal sigmas plus a joint covariance. **It is not the model's error against the
reference.** The fitted values are effective parameters for one molecule, method and distance range (a coupling of ~4 eV is not a physical
valence-bond coupling); whether they transfer is tested below, and they mostly do not. A fit can also take a Gaussian **prior** on any parameter
(`FitSettings.prior`, one extra residual `(value − mean)/σ` each, with the data's misfit reported separately from the prior's penalty).

**Transfer** (`substrate.transfer`, `python -m substrate transfer`). Three tests of whether parameters fitted to reference A say anything about
reference B, none of which is the fit's own uncertainty. (1) *Cross-prediction*: A's parameters, unchanged, on B's grid against B's real energies;
the only freedom is a constant energy shift (each calculation's zero is arbitrary; the shift is fitted on B's points, which flatters the transfer).
The diagonal is each reference's own calibration, and two no-model baselines sit beside it (the best constant energy, and no barrier at all). (2)
*Downstream*: the real chain's rate and isotope effect for B at the distance where B's reference barrier is 0.15, 0.4 or 0.8 eV (found by interpolating
B's own barriers), against the same chain run on A's parameters over B's grid and B's atom mass; a chain that refuses (the barrier is below the
zero-point energy) is reported as a refusal, not a number. (3) *Few-shot learning*: fit B from only k of its heavy-atom distances (the even-numbered ones;
the five odd-numbered ones are never trained on), from scratch or with A's values as a Gaussian prior of relative width s (0.1, 0.3, 1: three widths,
none tuned on the test distances); k = 0 is A's parameters alone, scored with the best constant shift on the held-out points. All eighteen fits define the
coupling at one common distance (2.7 Å) so parameters share variables. The references are `experiments/references/*.yaml`: for the nine symmetric ones a 21 × 11
rigid-group grid each (proton ±0.7–0.9 Å, heavy-atom distance 2.1–3.3 Å depending on the molecule), mirror-symmetric; for the three water–ammonia ones 25 × 11
with every proton position computed (proton ±1.0 Å, 2.4–3.4 Å), and 23 × 11 (proton ±0.9 Å) for the six methyl-derivative ones (2.4–3.2 Å methanol–water, 2.5–3.4 Å ammonia–methylamine). A reference file may declare its own fit window (`calibration: {window_ev: ...}`; 1.5 eV by default,
2.5 eV for all three asymmetric molecules, the last two chosen by the rule: the smallest of 1.5, 2.0, 2.5 or 3.0 eV that reaches every barrier top), and each target is scored over its own window; barriers are measured from the donor-side well. On this machine, under load, a scan of 133
energies took 1–2 min at HF, ~2 min at MP2 and 7–14 min at B3LYP (41 minutes for the eight symmetric ones that were not already cached); the 275 energies of a
water–ammonia scan took 2.5 min at HF, 2.8 at MP2 and 46 at B3LYP (with other jobs running), a methanol–water one 2.4, 5.6 and 27.5 min, an ammonia–methylamine one 2.9, 5.9 and 41 min. The few-shot section takes a symmetric parent as a prior for each methyl derivative as well.

**Electronic → quantum** takes the 1D slice and applies the proton mass. Errors if the slice has fewer than two wells or fewer than 15 points;
warns on a small electronic gap, a short table, or a spacing above 0.1 Å (coarser than a proton's zero-point width, so the nuclear levels start
to depend on the spline). **Electronic → molecular** takes the whole 2D surface and applies atomic
masses (the surface is mass-independent, so H and D share it); it warns if the gap *along the relaxed path* (not the
worst-case corner of the table) is small. Gap and table-margin thresholds are heuristics.

**Molecular** (`molecular.stationary_points_tst`). Bicubic spline of the table; the relaxed profile (minimum over `R` at each
`x`) gives the reactant, product and transition-state guesses, refined by Newton on the analytic gradient to true stationary
points (minima must have no imaginary mode, the TS exactly one). Normal modes: Hessian in `(x, R)` → Cartesian → mass-weighted →
translation projected out. Free energies include zero-point energy:
`G = E + Σ [ħω/2 + kT ln(1 - e^(-ħω/kT))]`. Rates: `k = (kT/h) exp(-ΔG‡/kT)`, with detailed balance exact by construction.
Approximations: harmonic modes (an O–H stretch is strongly anharmonic), vibrations only, **no tunnelling**, no recrossing.

**Molecular → reaction** errors if the free-energy barrier is not positive (zero-point energy has removed it), warns if it is
under 5 kT, and warns when `ħω‡/kT > 2` (tunnelling matters and classical TST underestimates the rate: ~12 at 300 K here).

**Quantum.** `H = -(ħ²/2m)∂² + V(x)` by finite differences. Levels per well come from half-domains truncated at the barrier top. Wells
are the two lowest minima with at least 5 meV of prominence, so numerical noise is not mistaken for chemistry. Barrier
transmission uses a stable amplitude recursion accurate to T ≈ 1e-200, where `1-|R|²` would round to 0.

**Quantum → reaction:**

```
k_f = (1/Q_L) Σₙ νₙ P(Eₙ) exp(-(Eₙ-E₀)/kT)      k_r = k_f · Q_L / Q_R   (detailed balance)
```

Approximations: single reaction coordinate; bath-assisted incoherent (sequential) tunnelling; above-barrier flux neglected;
Dirichlet-truncation error bounded by the tunnel splitting. Errors if the barrier is below the zero-point energy; warns if it
is under 5 kT above the reactant ground state.

**Reaction → biophysical** (`reaction_to_enzyme.chemical_step`). The two-state reaction becomes the chemical step `ES ⇌ EP`.
Required context: `context.k_on` (1/(M s)), `context.k_off`, `context.k_release` (1/s), `context.substrate` (M); optional
`context.product` and `context.k_on_product` (enables product rebinding, and with it the Haldane relation and a reversible
enzyme). Errors name every missing context parameter. Warns if an association rate exceeds the diffusion limit (1e10 1/(M s)), or
if the chemical rate is within 10× of the transition-state frequency limit kT/h (the step is nearly barrierless and a constant-rate
description is doubtful). Approximations: the chemical step is the *isolated model reaction* (a real enzyme's electrostatic
environment, which changes barriers, is not represented); binding and release are supplied, not computed.

**Biophysical** (`biophysical.enzyme_cycle`). A general cyclic-mechanism solver. For given substrate and product
concentrations the steady state is the stationary vector of the cycle's rate matrix (binding steps carry the ligand
concentration), computed with the subtraction-free Grassmann–Taksar–Heyman algorithm. Reports turnover per enzyme, state
populations, and, from the exact hyperbola `1/v` linear in `1/[S]`, `k_cat`, `K_M` and `k_cat/K_M`; for a reversible enzyme also
`kcat_reverse` and `KM_product` from the same cycle run backwards. (A mechanism that is not hyperbolic, e.g. substrate inhibition,
is refused rather than misreported.) Also: `K_d` and the binding free energy, the chemical and Haldane overall equilibrium
constants, and flux-control coefficients `∂ln k_cat/∂ln k_i` for every rate (they sum to 1).
Approximations: a single enzyme at steady state, fixed concentrations (no depletion), Markovian constant-rate steps, no
cooperativity, conformational heterogeneity, membranes or diffusion.

**Biophysical → biological** (`biophysical_to_pathway.enzyme_in_pathway`). The enzyme's constants become its rate law in the
pathway: `V_f = k_cat E_T`, `K_s = K_M`, and for a reversible enzyme the thermodynamically consistent
`v = (V_f S/K_s − V_r P/K_p) / (1 + S/K_s + P/K_p)`, which **is exactly the cycle's steady-state turnover** (verified to 1e-9 at
random concentrations), so no information is lost in the translation. An enzyme without product rebinding becomes irreversible
Michaelis–Menten. Required context: `context.enzyme_total`, `context.external_substrate` (M), `context.transport_rate` (1/s),
`context.drain_vmax` (M/s), `context.drain_km` (M). Warns if the enzyme exceeds 0.1 × K_M (the quasi-steady-state rate law is
then questionable).

**Biological** (`biological.metabolic_network`). A general metabolic-network engine: species, reactions with stoichiometry, and a
small rate-law library (`exchange`, `michaelis_menten`, `reversible_mm`, `first_order`, `constant`); a concentration reference names
either an internal metabolite or a fixed boundary pool. It follows the system's own dynamics until it settles and then polishes
with a Newton-type solve, checks that the steady state is **stable** (refuses otherwise), reports concentrations, fluxes, the slowest
relaxation time and a time course, and computes **flux-control coefficients** `C_i = ∂ln|J|/∂ln a_i` (summing to 1, for forward and
reversed flux alike; omitted at exact equilibrium where `J = 0`). Refuses, quickly and with the reason, a network that never settles
(a bounded integration budget), one whose "steady state" exceeds any physical concentration (100 M), and one whose integrator fails.
Approximations: a well-mixed compartment, deterministic, quasi-steady-state enzyme laws, no regulation, expression or growth.

## Validation

366 tests (48 need PySCF and skip without it). The 318 that do not take 1.5–4 min here depending on what else the machine is doing (90 s on a quiet one, 230 s on a busy one; the calibration fits, transfer fits and ensembles dominate), the whole suite 2–6 min with the real-chemistry energies cached (a cold WSL start adds a minute or two, and a first run computes a few hundred single points: ~8 more minutes unloaded, ~20 under load); they use closed-form results and independent calculations rather than the code agreeing with itself:

- **Calibration, against known truth.** On a synthetic reference built from known parameters: a noise-free surface is recovered exactly
  (every parameter to 0.2%, fit error 1e-5 eV); with 5 meV of noise, across independent noise realisations, the fitted values land within a few
  of their *reported* sigmas of the truth (max |z| < 4, RMS < 1.6), so the uncertainties are neither over- nor under-confident; the closed-form
  model used in the fit equals the engine's diagonalisation to 1e-10; a reference that needs a parameter beyond its bound is flagged (and the
  fit visibly worse); a symmetric reference fixes the offset and an asymmetric one recovers it; leave-one-out on an exact model predicts an unseen
  distance exactly; the 1D model cut from the 2D one coincides with the 2D slice to 1e-9 up to a constant; joint draws reproduce the covariance
  matrix (to 10%) while leaving other parameters independent; truncated draws respect the floors; malformed covariances are refused; ignoring the
  correlations inflates the spread several-fold.
- **Calibration, against real chemistry.** Fitted to the Hartree–Fock/6-31G* Zundel surface: RMSE ≤ 0.015 eV and worst error < 0.04 eV over the
  fitted window; the same number of wells at every distance; barrier errors < 0.03 eV; held-out distances predicted to 0.02 eV. Downstream it
  reproduces the real chain's rates to within 0.6–1.6× across 2.6–3.0 Å and the isotope effects to 30%, where the uncalibrated model was off by
  more than three orders of magnitude; the 1D and 2D models agree; the calibrated quartic double well reproduces the real slice rates to 25%;
  and the test that the fit uncertainty is smaller than the model's real error is itself an assertion, so the caveat cannot rot.
- **Transfer, against known truth** (`tests/test_transfer.py`, synthetic references whose parameters are known). The comparison itself has closed-form
  checks: an energy-zero offset costs nothing, a ±20 meV zero-mean wiggle scores exactly 20 meV, points outside the window are not scored, the worst error
  and barrier errors keep their sign and the mean is of magnitudes, and a model with no barrier misses exactly the barriers the reference has. A
  calibration predicts its own reference to 1e-5 eV (an asymmetric one too, through its fitted offset); the error grows with how different the reference is;
  a weaker coupling overestimates every barrier; the matrix is keyed and labelled source-then-target and is not symmetric. A prior at the truth predicts unseen
  distances from one distance; a wrong prior hurts in proportion to how tightly it is held and is overruled by enough data; a prior defined at another reference
  distance is read in its own variables; the prior's penalty is not counted as misfit; a prior lets a fit proceed where the data alone cannot (and where it
  cannot, the refusal is recorded, not scored). The learning-curve score equals an independent hand fit scored by hand. The rate comparison runs the model chain on
  the *target's* grid, heavy-atom mass and both isotopes (checked by spying on the systems the chain receives). A weaker-coupling calibration gives a slower
  rate and a larger isotope effect at every barrier, but only 0.4–0.55× slower for a barrier ~0.09 eV higher where classical TST would give 0.03×: the quantum
  route is tunnelling-dominated here and much less sensitive to the barrier (my first test assumed TST and failed against the measurement).
- **Transfer, against real chemistry** (`tests/test_transfer_real.py`; the three Hartree–Fock references, ~3.5 min on a cold cache). Each template is recorded
  on its solved surface with its atoms and charge; every surface is symmetric and its barrier grows monotonically with distance; each own fit is within a few
  tolerances; the X–H length and Morse width are physical; parameters fitted to one molecule predict another with ≥10× the own-fit error and a barrier error
  >0.08 eV in every one of the six directions; and the rates agree for a calibration's own molecule (0.5–2×) and miss by ≥5× in every other case where both
  chains give one. For the asymmetric ion (the HF surface, ~19 min on a cold cache under load, 3 min unloaded): the surface is asymmetric with the proton on the
  nitrogen, one well up to 2.7 Å and two from 2.8 Å with the oxygen-side well 0.9–2.0 eV up; the fit with the declared window follows it (RMSE < 0.06 eV, every well count right, every
  barrier within 0.15 eV, offset below −1 eV) and with the offset forced to zero is more than five times worse and loses a well; the declared window is what lets the fit see the
  barrier (the 3.4 Å barrier error is larger at the default window); leave-one-distance-out predicts unseen distances; and symmetric parameters cannot predict the asymmetric surface, nor
  the reverse (≥ 0.3 eV, ≥ 2 wrong well counts, in all six directions with the three HF references). For the methanol–water ion (HF; ~4 min on a cold cache): the surface is asymmetric
  with the proton on the methanol, one well up to 2.56 Å, two from 2.64 Å, the donor-side well 0.3–0.9 eV up; the fit with its declared window follows it (RMSE < 0.07 eV, every well count right, every
  barrier within 0.06 eV, offset between −2.5 and −1.0 eV) and is several times worse without the offset; leave-one-distance-out predicts unseen distances; and transfer degrades with chemical
  distance (Zundel → methanol–water 0.15–0.35 eV, Zundel → water–ammonia above 0.4, methanol–water → water–ammonia smaller than Zundel → water–ammonia). For the ammonia–methylamine ion (HF; ~3 min unloaded, ~18 under load): asymmetric with
  the proton on the methylamine, one well up to 2.59 Å, two from 2.68 Å, the donor-side well 0.2–0.5 eV up; the fit follows it (RMSE < 0.055 eV, every well count right, every barrier within 0.1 eV), is several times worse without the
  offset, and its offset is the smallest of the three asymmetric ions' in the order of their proton-affinity gaps; leave-one-distance-out predicts unseen distances; and the symmetric parent is its nearest neighbour (0.1–0.25 eV in
  both directions, but more than 3× its own error), every change of heavy atom being at least 1.5× farther. These record findings, not hopes: if a model change makes parameters transfer, they fail, which is a prompt to look, not to loosen a threshold.
- **The GUI** (`tests/test_gui.py`, 35 tests over real HTTP on a free local port). Listing and experiment endpoints; paths that try to leave `experiments/` (parent, encoded, absolute, another extension) are refused; a run returns every stage with a lineage in which each hop's input hash is the previous hop's output hash; the plotted curve is the engine's own array (to the transport rounding); editing the coupling lowers the barrier and does not leak into the next run; an ensemble attaches spread; a refusal (O···O equilibrium 2.3 Å) keeps the electronic stage and its plots and names the molecular stage that refused; bad edits (an unknown name, a non-finite number, a negative σ) and oversized ensembles are refused before anything runs; jobs report messages and errors; a foreign Host or Origin, a non-JSON, malformed, oversized or traversal request is refused; the page's files are served under the policy with no inline script, and its two scripts parse (when node is present). On synthetic references the transfer payload has every pair, recovers each reference exactly, is saved, reloads, and is invalidated when a reference file changes; the rate comparison is ~1 on the diagonal; the learning-curve payload says why a one-distance fit is absent. **Not tested automatically: how the page looks and behaves in a browser.** I checked that by hand (both themes, hover, all five transfer views) and found five defects that way, now fixed: nested lists rendered as "[object HTMLButtonElement]", truncated parameter labels, a colour bar that overflowed its card, a stray "null" printed by `replaceChildren`, and reference points drawn in a status-like red.
- **Molecule templates** (`tests/test_molecules.py`). Every template is valid for the engine; every symmetric one maps onto itself with donor and acceptor exchanged (the
  symmetry `mirror_symmetric` relies on: a reflection across a 45° plane for the Zundel ion, a 60° turn or inversion for the ammonium dimer); the water–ammonia ion's
  bond lengths, angles, flank sizes, twist and reduced-mass entry are checked against its parameters, as are the methanol–water ion's (O–C, C–O–H, H–C–O, C–H, the staggered
  methyl with one C–H anti to the O–H, the independent O–H options, `twist` and `rotor` moving only what they should) and the ammonia–methylamine ion's (the same, with the methyl anti to the hydrogen bond and independent N–H and polar-angle options); the staggered-methyl builder both share is tested on its own; none of the three asymmetric ions has any exchange operation, and against
  an analytic asymmetric surface the engine computes the full grid for each, refuses a mirror-symmetry claim, and reproduces the formula at every point; the ammonium dimer's
  N–H lengths, angles and 120°/60° staggering are checked against its parameters; groups move rigidly with their heavy atom in the scan; names resolve, record
  themselves, and are idempotent; the engine records the resolved molecule and refuses an unknown name as a validation error.
- **Real quantum chemistry, against physics.** Through the real PySCF bridge: H₂/cc-pVTZ comes out above the Hartree–Fock limit
  (−1.133629) and within 1 mHartree of it, and the H atom (open shell) above the exact −0.5, both as the variational principle requires;
  the energy is invariant under rigid rotation and translation to 1e-7; MP2 lies 0.1–0.3 Hartree below HF for water; an unknown theory
  is an error. On the Zundel ion: E(+x) = E(−x) to 1e-8 Hartree (the exchange symmetry of the staggered ion, a two-fold axis), the proton is shared (one well) at 2.3 Å and
  localised (two wells) at 3.0 Å, and the barrier rises monotonically with O···O distance.
- **Real quantum chemistry, plumbing.** A *fake* program returning an analytic energy surface (written out independently of the engine, with its
  own unit constant) lets the engine be checked against a formula: the surface equals the formula at every grid point, so a mistake in the
  (x, R) → coordinates mapping, in which atoms move with which heavy atom (a rigid-group penalty makes a stray atom cost energy), or in the
  Hartree → eV conversion shows up as a mismatch. With a fake that returns the *model* engine's energies, the whole molecular route
  reproduces the model engine's barrier, frequency and rate to 1e-6: the real-chemistry engine is a drop-in for the model one. Mirror
  symmetry gives the same surface with about half the calculations, and a *false* symmetry claim is caught. Reruns compute nothing;
  duplicates are computed once; unconverged energies are not cached; a program crash is not swallowed by an ensemble.
- **Biological.** A one-metabolite network's steady state and flux match the closed-form quadratic to 1e-8; the control coefficients
  match differentiating that closed form, and sum to 1; a pathway at its enzyme's equilibrium has zero flux and reverses on either side
  (with finite control that still sums to 1); the time course settles onto the steady state found by root-finding (two independent
  methods); the pathway's reversible rate law reproduces the biophysical cycle's exact turnover and the Haldane relation holds; the
  isotope-propagation regimes above are asserted. Instability is tested by registering a deliberately self-amplifying law.
- **Biophysical.** `k_cat` and `K_M` match the hand-derived closed forms to 1e-8 over four parameter sets; the steady state agrees
  with an **independent mass-action ODE simulation** of the full bimolecular network; at the Haldane ratio the net flux vanishes and
  reverses on either side; control coefficients obey the summation theorem; limiting regimes (slow release → isotope effect 1.00, fast
  release → the full 50× for a 50× effect). The stationary-distribution routine is checked against linear algebra and exact rational
  arithmetic for rates 10 orders of magnitude apart.
- **Electronic.** The surface matches the closed-form two-level energy to 1e-12; the symmetric crossing is lowered by exactly the
  coupling; the zero-coupling limit is the lower envelope; well depth follows second-order perturbation theory; each column of the 2D
  surface equals the 1D engine at that distance (which also pins down the axis order); the tabulated nuclear engine reproduces the
  analytic one to 1e-6.
- **Molecular.** Normal-mode eigenvalues match a finite-difference *Cartesian* Hessian for unequal masses and a cross term; for a
  surface stiff in `R`, frequencies match the analytic reduced-mass result; the harmonic ZPE of the O–H bond agrees with the
  *anharmonic* quantum levels of the same bond within 5% and lies above them, as anharmonicity requires; the rate is recomputed
  independently from the reported frequencies; detailed balance holds; the semiclassical KIE (~10) is in the textbook range.
- **Quantum.** Transmission vs the analytic rectangular barrier (~1e-14, including T ~ 1e-211); harmonic-oscillator levels; parity of
  the symmetric doublet; the tunnel splitting from **diagonalisation** vs `(ħω/π)√P` from the **scattering** code agree within ~6% for
  0.4–0.8 eV barriers (the test allows 15%; the gap grows to ~11% at 1.0 eV).
- **Cross-scale physics.** Heavy-atom relaxation lowers the barrier on the *same* surface; isotopes share one electronic surface;
  rate rises with temperature; the barrier disappears in stages with a refusal at each gate; routing is kind-aware and ambiguity is an
  error; context parameters arrive unchanged at every scale; provenance forms an unbroken hash chain across all nine stages; ensembles
  are seed-reproducible.
- **The tests can fail.** Deliberately injected bugs (a 1% coupling error, a 1 meV surface shift, no mass weighting, no zero-point term,
  a wrong coordinate-transform row, an inverted K_d, the wrong state's population, a broken stationary-distribution step, swapped
  chemical rates, context parameters not carried, a wrong Haldane factor, a flipped exchange-law sign, a dropped reverse-rate sign, a
  missing enzyme concentration, the wrong K_M for the product, a control analysis perturbing the wrong reaction, the stability check
  removed, reverse kinetics returning the forward values, and for the quantum-chemistry engine a wrong Hartree→eV constant, a group not
  travelling with its oxygen, an unverified symmetry claim, a mirrored half in the wrong order, the proton on the wrong side, energies not
  referenced to the minimum, and the empty-cache bug below) are each caught.
- **Mutation testing of the transfer code.** 43 one-line mutants of `transfer.py` (flipped signs, a median for a mean, a window doubled, labels
  swapped, the wrong distance or mass handed to the chain, the split or the prior width changed, and so on): 31 were caught at first and 12 survived. Ten of
  those were real gaps (the worst error, the sign of barrier errors, missed barriers, three label/key mix-ups, the order of distances in an interpolation, a prior
  read in the wrong variable, a prior with no width, the template label), now tested and caught on a rerun; the other two are equivalent mutants (barrier
  and well positions do not change with the energy zero; two dropped dictionary entries that are passed separately). Twelve mutants of the prior code in
  `calibration.py`: 11 caught, 1 survivor that changes only the optimiser's starting point (not shown to change any result). A third pass over the declared-window, hint-parser and water–ammonia template code: 20 mutants, all caught once one test was tightened. The one that first survived (the ammonia unit's twist) showed my test did not pin the geometry, and looking at it the geometry was less staggered than my docstring said (nearest hydrogens 28° apart; a 0° twist would give 58°). Computing HF energies at four twists shows the choice moves the energy by at most 0.5 meV (the fit tolerance is 10 meV), so the surfaces stand; the docstring is corrected and the twist is now an option. The methanol–water template and the spread-ratio display were mutation-tested too: 14 mutants of the template, 12 caught at once, one equivalent (swapping the hydroxyl hydrogen and the carbon to opposite azimuths gives the mirror image of the same arrangement, which has the same energy), and one that showed an option (the methanol O–H length) that no test exercised, now tested; of 3 mutants of the spread-ratio floors, 2 survived until tests were added for a small but real value and a pair seven decades apart. The ammonia–methylamine template: 16 mutants (every default, the twist and rotor, the anti direction, the group list, the mass, each flank's own bond length and angle, and the staggered-methyl builder both methyl ions now share), all caught on the first pass. Moving that builder out of the methanol–water template was checked against the energy cache, which is keyed by geometry to 1e-8 Å: all 828 methanol–water energies were hits, so no coordinate moved.
- **Real bugs the suite and its own probes found in my code:** an empty explicitly-passed energy cache that was falsy (it defines `__len__`) and
  so silently replaced by the user's default cache, which put test energies into the real cache until the regression test caught it (the
  stray entry is removed); a scan-size rule of "≥ 50 points" tuned for dense model grids that rejected a legitimate 21-point ab initio slice
  (now a resolution check tied to the proton's ~0.1 Å zero-point width); a slice whose excited state missed the O···O shift (wrong electronic
  gap); a parameter doing two jobs (coupling reference and slice distance); a numerical-stability bug in the enzyme engine (rates 10
  orders of magnitude apart lost ~6 digits; fixed with GTH and by measuring turnover at product release); control coefficients that were
  NaN for a reversed pathway; an integrator failure that escaped as a raw `RuntimeError` instead of a refusal; and a network that sent
  the steady-state search into minutes of integration, or "converged" to thousands of molar, which is now refused in seconds with the
  reason. Running the example also showed that the caption I first wrote for the control-coefficient prediction ("follows the trend")
  was wrong at the point where it matters (predicted 1.02, true 3.81), which is why the example prints the table rather than only
  asserting a claim. Calibration added its own: a first tolerance of 10% of the energy let the barrier region drift 0.07 eV (a 2.5× rate
  error); a 3 eV cap on the coupling silently became the answer (the data's optimum is 4.1 eV); a Gaussian draw of a poorly-determined
  well depth went negative in 14% of ensemble draws and whole runs were being discarded (now truncated at the physical floor); and a fit on
  the model's own dense default grid, plus leave-one-out over every one of its 91 distances, ran for many minutes (the fit cost scaled with the
  grid, the validation with the number of distances; now thinned to ~1500 points and 12 warm-started folds, 3.7 s, with identical answers).
  The asymmetric references added: the learning curves gave an asymmetric source no prior on its energy offset (the parameter that matters most for it), which I
  found by reading the output; a parameter pinned at ~0 produced a ratio of 10²¹ in the spread table; and the 1.5 eV window every earlier fit used turned out to
  be a decision the data can overrule (it hides the asymmetric barrier), which is why a reference file can now declare its own.
  The second asymmetric molecule added: a spread-table cell of "616×" for three values all pinned within 1e-15 of zero (the guard compared against the largest, not against zero), and my own
  claim that the rates "follow the same order" as the energies, which the numbers did not support (they are far noisier) and which I replaced with the numbers.
  The transfer work added: the held-out score reported only fixed energy windows (0.5, 1, 1.5 eV), so a custom window raised a `KeyError` (found by a test
  that used one); my claim that the Zundel template is "S4-symmetric" was wrong (a staggered pair of waters maps onto itself through a two-fold axis, not a
  rotation-reflection; the engine's numerical symmetry check had been right all along, and the docstring, comments and this README are corrected); and two
  test thresholds I guessed (a rate ratio below 0.2, a held-out error above 0.05 eV) failed against measurement and were replaced by what was measured.
  Mutation testing of the new code (below) found ten untested behaviours, now tested.

- **One unexplained intermittent failure.** In a single full-suite run one test failed and I did not capture which; it did not recur in the
  ten full or partial runs that followed (and a 40-call stress test of the WSL bridge had no failures). The only wall-clock assertion in the suite,
  which could have been the cause on a machine that is sometimes 2–3× slower, was replaced by a deterministic bound on work done, but I
  cannot show that was it. If you see an intermittent failure, run `pytest -rf --tb=short` repeatedly and keep the output. It happened a second time at the end of the methanol–water work, in one full run among the many since (I did not count them), and this time I have the test's name: `tests/test_pipeline.py::test_fingerprint_is_deterministic_and_sensitive` (I did not keep the assertion that failed). It did not recur in 25 reruns of its module, six more full-suite runs, or 300 identical pipeline runs under heavy CPU contention (which gave one fingerprint and bitwise-identical values, so the explanation I first suspected, last-bit solver noise tipping the 12-digit rounding that fingerprints use, has no support). A rounding-based hash can in principle flip that way, which is the only weakness in the design I can point to; I have not changed it on a guess. To catch it, run `pytest -rf --tb=long` repeatedly and keep the output.

## Read the numbers honestly

The model engines' parameters are a toy in a plausible range, **not** fitted to a molecule, an enzyme or an organism, and the real
quantum-chemistry run proves it: at the same O···O distances the model's barriers are 2–12× larger than Hartree–Fock/6-31G* (the gap
narrows as the distance grows). That is what calibration repairs, for one molecule at one level of theory: the calibrated model is accurate
where it was fitted (rates within ~1.3× of the real chain over 2.6–3.0 Å), but its parameters are effective rather than physical, **they do not
carry to another molecule (worse than a constant for 22 of 72 reference pairs) or from Hartree–Fock to B3LYP/MP2 (rates off by ~550× at the median)**,
and **its stated uncertainty covers only the parameters, not the part of the real
surface a fixed functional form cannot represent** (the model's rate error, up to 28%, exceeds the fit's 6% spread). The transfer study has limits of its
own: eighteen references (nine symmetric ones over three molecules, and three asymmetric molecules at three methods each, with proton-affinity gaps of ~0.47, ~0.65 and ~1.7 eV; two of the
three are one-methyl derivatives of a symmetric parent, so the 0.27–0.38 eV per eV observation rests on two derivatives, and there is no MP2 ammonium dimer), three methods and two basis sets; the asymmetric fits' energy offset is an effective parameter (water–ammonia −2.1, −4.7 and −3.5 eV, methanol–water −1.7, −3.6 and −2.6 eV, ammonia–methylamine −0.8, −1.5 and −1.1 eV at HF, B3LYP
and MP2, so it varies 1.8–2.2× between methods; it is ordered as the proton-affinity gaps but its ratios between the molecules, 1.3–1.4, 2.0–2.4 and 2.5–3.2, do not track the ratios of the gaps, ~1.4, ~2.6 and ~3.6), and their fit window is a declared decision, not a derived one (2.5 eV misses the HF barrier top at 3.4 Å, 2.7 eV; at
3.5 eV the fit's error over its own, larger window is 0.059 eV); all scans are rigid, with the flank geometry frozen at one template for every method (no method relaxed its own
structure); the energy zero is fitted on the target when transferring, which flatters the transfer; six of the nine symmetric fits leave the O···O equilibrium distance
on its 4 Å bound (and eight of the nine asymmetric ones leave the coupling's decay on its lower bound, 0) and the O···O parameters are not identified by the data (widening the bound changes no cross-prediction RMSE by more than 0.008 eV and no mean barrier error by more than 0.057 eV, median 0.002, so
the conclusions do not depend on it, but those two parameters mean nothing individually and the individual barrier errors of the transfers carry that much slack); the rate comparison is at one temperature and at distances chosen from
the target's own barrier; the learning curves use one fixed split and three prior widths. The real-chemistry
results are themselves the output of stated approximations: a
single level of theory and a small basis (Hartree–Fock/6-31G*, no electron correlation, no basis-set convergence study), a **rigid**
scan (the water-like groups' bond lengths and angles do not relax), a collinear O–H–O axis, gas phase, and a HOMO–LUMO gap standing in
for the excitation gap. They are not the Zundel cation's true rates. The chemical step inside the enzyme is the isolated model reaction: real enzymes change barriers
through their electrostatic environment, which is not represented, and the binding, release and pathway parameters are free inputs.
So the higher-scale outputs show *how* lower-scale chemistry is expressed, attenuated or hidden in measurable quantities, which is
what the architecture is for, not what any real system does. The biological module is a single well-mixed compartment at
steady state with no regulation, expression or noise.

The GUI is a front end to the engines and inherits all of the above; it adds limits of its own. It runs only the experiment files in `experiments/` (no authoring or uploading), serves one user on one machine, cannot cancel a running job (a long quantum-chemistry run goes on in the background), edits only scalar inputs, and ensembles of a slow experiment re-run on every edit unless you turn that off (it turns itself off after a run that took over six seconds). A job the server loses on restart is lost, though a finished transfer study is saved and reloaded. The quantum-chemistry experiments need PySCF; without it the GUI says so and reads only energies that are already cached.

The rate is extremely sensitive to the O···O distance in this model: shortening the O···O equilibrium distance from 2.70 to 2.66 Å
lowers the barrier by 0.18 eV and raises the TST rate ~1000-fold. Rates near 10¹² s⁻¹ push the sequential-tunnelling assumption, which
cannot be checked without a bath. Control coefficients are local; see the note on non-locality above.

## Adding the next scale or model

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
