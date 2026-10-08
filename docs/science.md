# The implemented science

[← back to the README](../README.md)

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
Approximations: two valence-bond states only, one Morse curve for both bonds (or, for a pair whose two bonds differ, a curve of its own for the acceptor's:
`acceptor_morse_depth`, `acceptor_morse_alpha`, `acceptor_morse_r_eq`, any of which that is given replaces the donor's value on the acceptor side), collinear geometry, no environment,
ground state only. The coupling's `reference_distance` only places the coupling and may lie outside the scan range; the `scan_distance` may not.

**Real quantum chemistry** (`electronic.qc_scan_2d`). The same observables from real energies. A molecule with its donor, proton and
acceptor on a z axis is described in the experiment's `structure`; every other atom belongs to a rigid group that travels with the donor or
the acceptor. For each proton displacement `x` and heavy-atom distance `R` on a grid, the engine builds the geometry, asks the program for a
single-point energy (`hf`, `mp2`, `ccsd`, `ccsd(t)` or `dft:<functional>`, any basis; open-shell handled by the unrestricted variants, coupled cluster closed-shell only and refused otherwise; MP2 and coupled cluster are all-electron; a coupled-cluster point costs minutes, so those jobs travel in chunks of six), and assembles
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
valence-bond coupling); whether they transfer is tested in [findings.md](findings.md), and they mostly do not. A fit can also take a Gaussian **prior** on any parameter
(`FitSettings.prior`, one extra residual `(value − mean)/σ` each, with the data's misfit reported separately from the prior's penalty).
A pair whose two bonds differ (Cl–H 1.27 Å against F–H 0.92 Å) cannot be placed by one Morse curve; `FitSettings(bonds="separate")`, or `bonds: separate` in an experiment file's
`calibration:` hint, or `calibrate --bonds separate`, gives the acceptor bond a curve of its own (depth, width, equilibrium length). It is refused for a symmetric surface and is not the default.

**Transfer** (`substrate.transfer`, `python -m substrate transfer`). Three tests of whether parameters fitted to reference A say anything about
reference B, none of which is the fit's own uncertainty. (1) *Cross-prediction*: A's parameters, unchanged, on B's grid against B's real energies;
the only freedom is a constant energy shift (each calculation's zero is arbitrary; the shift is fitted on B's points, which flatters the transfer).
The diagonal is each reference's own calibration, and two no-model baselines sit beside it (the best constant energy, and no barrier at all). (2)
*Downstream*: the real chain's rate and isotope effect for B at the distance where B's reference barrier is 0.15, 0.4 or 0.8 eV (found by interpolating
B's own barriers), against the same chain run on A's parameters over B's grid and B's atom mass; a chain that refuses (the barrier is below the
zero-point energy) is reported as a refusal, not a number. (3) *Few-shot learning*: fit B from only k of its heavy-atom distances (the even-numbered ones;
the five odd-numbered ones are never trained on), from scratch or with A's values as a Gaussian prior of relative width s (0.1, 0.3, 1: three widths,
none tuned on the test distances); k = 0 is A's parameters alone, scored with the best constant shift on the held-out points. All twenty-one fits define the
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
