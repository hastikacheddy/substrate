# Read the numbers honestly

[← back to the README](../README.md)

What this project can and cannot support, in one place. The short version: the software is carefully tested and its statements about itself
are measured; the chemistry is a *testbed* for a methodological question (do fitted model parameters carry over?), built on rigid, collinear,
gas-phase scans with small basis sets. None of it is a prediction of a real system's kinetics.

## What the model engines are

- The model engines' parameters are a toy in a plausible range, **not** fitted to a molecule, an enzyme or an organism. The real
  quantum-chemistry run proves it: at the same O···O distances the toy model's barriers are 2–12× larger than Hartree–Fock/6-31G* (the gap
  narrows as the distance grows).
- Calibration repairs that for one molecule at one level of theory: the calibrated model is accurate where it was fitted (rates within ~1.3× of
  the real chain over 2.6–3.0 Å), but its parameters are **effective rather than physical**.
- **The stated uncertainty covers only the parameters**, not the part of the real surface a fixed functional form cannot represent (the model's
  rate error, up to 28%, exceeds the fit's 6% spread).

## The model is used as a flexible fit

- Twelve correlated parameters (asymmetric surface, separate bonds) fit a few hundred points to 0.01–0.06 eV. A good rmse shows flexibility,
  not that the model is right.
- Fitted couplings are 3–20 eV and fitted energy offsets are −0.8 to −5.4 eV for the asymmetric ions at HF (and run to the −10 eV bound for
  chloride–HF at B3LYP and MP2); real diabatic couplings are far smaller and the measured long-range well gaps are 0.4–1.7 eV. These are effective parameters. The offset of the chloride–HF ion sits on its −10 eV bound at B3LYP and MP2 and trades off against a
  coupling of 17–20 eV along a long valley (with the bounds moved out it reaches −33 eV for B3LYP at 0.002 eV better rmse, in a nearly singular
  fit): not a number to compare between ions.
- The bond lengths and widths *are* physical (within a few percent of the free diatomics), but pinning them there does not help: with the
  equilibrium length and width held at the diatomic's the median fit error goes from 0.031 to 0.119 eV and transfer is no better in absolute error
  ([findings](findings.md)). The parameters that decide the barrier (coupling, heavy-atom well, offset) are the ones that depend on the molecule
  and the method.
- Six of the nine symmetric fits leave the O···O equilibrium distance on its 4 Å bound, and ten of the twelve asymmetric ones leave the coupling's
  decay on its lower bound, 0. The O···O parameters are not identified by the data (widening the bound changes no cross-prediction rmse by more than
  0.008 eV and no mean barrier error by more than 0.057 eV, median 0.002), so the conclusions do not depend on them, but those parameters mean nothing individually.
- The shared Morse curve cannot fit a pair whose two bonds differ (chloride–HF); the separate-bond model does.

## The surfaces

- All scans are **rigid**: every atom other than the proton moves with its heavy atom as a frozen group, the flank geometry is frozen at one
  template for every method (no method relaxed its own structure), the axis is collinear, and the energies are gas-phase. At a fixed long
  heavy-atom distance a metastable well on the weaker-base side can exist that an unconstrained system would not have.
- Levels of theory: Hartree–Fock, B3LYP and MP2 with 6-31G* (cations) and 6-31+G* (anions), plus HF/cc-pVDZ for one ion. The "electronic gap" is
  the HOMO–LUMO gap, a proxy for the excitation gap.
- **How wrong are they?** A CCSD(T) benchmark of the bifluoride ion ([findings](findings.md)) shows that at F···F = 2.74 Å the study's
  surfaces miss the CCSD(T)/aug-cc-pVTZ barrier by +0.19 eV (HF), −0.19 (B3LYP) and −0.14 (MP2), about 10³ in a transition-state rate at
  300 K. The 6-31+G* basis alone lowers every barrier by 0.09–0.14 eV. Over the whole surface HF/6-31+G* looks closest to the benchmark only
  because two errors partly cancel. That benchmark is one symmetric ion with six distances; the basis-set limit is not reached.
- They are not the Zundel cation's true rates.

## The transfer study's design

- Twenty-one references in the study (nine symmetric ones over three molecules, four asymmetric molecules at three methods each; the
  three chloride–HF surfaces are built and tested but not yet in the study). That is **seven molecules, not twenty-one independent samples**: the
  methods of one molecule are strongly correlated, so the pair counts (420 pairs, 89 worse than a constant guess) describe the set, not a
  population.
- The measured proton-affinity gaps (NIST WebBook) are 0.44 ± 0.09 (an anion), 0.47, 0.66 and 1.69 eV for the four asymmetric molecules in the study,
  and 1.66 eV for chloride–HF (not yet in it). **The surfaces do not all reproduce them**: with the separated fragments relaxed and the enthalpy at 298 K
  accounted for, the computed long-range gap is within about 0.1 eV of the measured one for ammonia–methylamine, methanol–water and water–ammonia at
  all three methods and for chloride–HF at B3LYP, but 0.22–0.33 eV too large for fluoride–methanol at every method (the 6-31+G* basis is the main cause:
  at B3LYP the gap falls from 0.83 to 0.66 eV in aug-cc-pVTZ) and 0.29 eV too large (HF) and 0.22 eV too small (MP2) for chloride–HF ([findings](findings.md)). Two of the four asymmetric molecules are one-methyl derivatives of a
  symmetric parent, so the observation that transfer follows chemical distance (0.27–0.38 eV of error per eV of proton-affinity gap) rests on two
  derivatives, and there is no MP2 ammonium dimer. It is an observation, not a law.
- The asymmetric fits' energy offset is an effective parameter: water–ammonia −2.1, −4.7 and −3.5 eV, methanol–water −1.7, −3.6 and −2.6,
  ammonia–methylamine −0.8, −1.5 and −1.1, fluoride–methanol −1.6, −3.7 and −3.1 eV at HF, B3LYP and MP2 (it varies 1.8–2.3× between methods).
  Across the three cations it is ordered as the proton-affinity gaps but its ratios (1.3–1.4, 2.0–2.4, 2.5–3.2) do not track the ratios of the gaps
  (~1.4, ~2.6, ~3.6), and the anion, with a measured gap of ~0.4 eV, has an offset as large as methanol–water's, because its surface's own long-range gap is
  1.1–1.2 eV (0.8–0.95 relaxed), larger than methanol–water's 0.8–0.9 eV: the offsets order with the *computed* gaps, to within the fit's uncertainty, and are 1.1–4.6× them.
- The fit window is a declared decision, not a derived one (2.5 eV misses the HF barrier top at 3.4 Å, 2.7 eV; at 3.5 eV the fit's error over its own,
  larger window is 0.059 eV).
- The energy zero is fitted on the target when transferring, which flatters the transfer. The rate comparison is at one temperature and at
  distances chosen from the target's own barrier. The learning curves use one fixed split and three prior widths.
- Two predictions failed and are recorded: the offset of fluoride–methanol was expected to be smaller than ammonia–methylamine's (it was made from the
  *measured* gap, which the study's surface does not have), and the nearest neighbour of chloride–HF was expected to be the bifluoride.
- The finding that parameters do not transfer is consistent with what the empirical-valence-bond literature already says (parameters are fitted
  per reaction). The contribution here is measuring it carefully, not discovering it.

## Higher scales

- The chemical step inside the enzyme is the isolated model reaction: real enzymes change barriers through their electrostatic environment, which
  is not represented, and the binding, release and pathway parameters are free inputs.
- The higher-scale outputs show *how* lower-scale chemistry is expressed, attenuated or hidden in measurable quantities, which is what the
  architecture is for, not what any real system does. The biological module is a single well-mixed compartment at steady state with no
  regulation, expression or noise.
- The rate is extremely sensitive to the O···O distance in the model: shortening the O···O equilibrium distance from 2.70 to 2.66 Å lowers the
  barrier by 0.18 eV and raises the TST rate ~1000-fold. Rates near 10¹² s⁻¹ push the sequential-tunnelling assumption, which cannot be checked
  without a bath. Control coefficients are local; see the note on non-locality in [findings](findings.md).

## The GUI

The GUI is a front end to the engines and inherits all of the above; it adds limits of its own. It runs only the experiment files in
`experiments/` (no authoring or uploading), serves one user on one machine, cannot cancel a running job (a long quantum-chemistry run goes on in
the background), edits only scalar inputs, and ensembles of a slow experiment re-run on every edit unless you turn that off (it turns itself off
after a run that took over six seconds). A job the server loses on restart is lost, though a finished transfer study is saved and reloaded. The
quantum-chemistry experiments need PySCF; without it the GUI says so and reads only energies that are already cached. Concurrent runs are safe but
queue at the ODE solver, which can integrate one problem at a time per process.
