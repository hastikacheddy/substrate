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
