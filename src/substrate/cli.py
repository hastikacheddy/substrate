"""`python -m substrate run <experiment.yaml>`: run an experiment and print a scale-propagation report."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .defaults import default_registry
from .errors import SubstrateError
from .experiment import load_experiment, read_spec
from .ir import Quantity, ScientificSystem
from .pipeline import Pipeline, RunResult


def fmt(q: Quantity) -> str:
    v = q.value
    if isinstance(v, np.ndarray):
        if v.size > 4:
            return f"array{list(v.shape)} [{v.flat[0]:.4g} ... {v.flat[-1]:.4g}] {q.unit}"
        return "[" + ", ".join(f"{x:.4g}" for x in v.ravel()) + f"] {q.unit}"
    s = f"{v:.4g}"
    if q.sigma is not None and np.ndim(q.sigma) == 0:
        if q.band is not None and q.sigma > 0.5 * abs(v):
            # the spread rivals the value (an exponentially sensitive quantity): a percentile band is honest, +/- is not
            s += f" (68% band {q.band[0]:.2g} to {q.band[1]:.2g})"
        else:
            s += f" +/- {q.sigma:.2g}"
    return s if q.unit in ("1", "count") else f"{s} {q.unit}"


def report(result: RunResult) -> str:
    lines = [f"EXPERIMENT  {result.experiment_id}", f"INPUT       {result.root.name}  [{result.root.fingerprint()}]", ""]
    lines.append("SCALE PROPAGATION")
    produced = [s for s in result.trace]
    for step, system in zip(result.steps, produced):
        rec = next(r for r in reversed(system.provenance) if r.step != "uncertainty")
        lines.append(
            f"  {step.scale_out.label:<22}{step.kind:<10}{step.component.name:<44}"
            f"{rec.seconds * 1e3:8.1f} ms  ok  [{rec.output_fingerprint}]"
        )
    if result.ensemble:
        e = result.ensemble
        lines.append(f"  ensemble: {e['n_ok']}/{e['n_requested']} samples valid over {', '.join(e['varied'])} (seed {e['seed']})")
    for system in produced:
        lines += ["", f"{system.scale.label.upper()}  {system.name}"]
        # an unsolved system (fresh from a translator) has no observables yet: show what it was derived with
        derived = {} if system.observables else {
            n: q for n, q in system.parameters.items() if q.source and q.source != "input"
        }
        for name, q in {**derived, **system.observables}.items():
            lines.append(f"  {name:<30}{fmt(q)}")
    warnings = result.warnings()
    lines += ["", f"WARNINGS ({len(warnings)})"] + [f"  - {w}" for w in warnings] if warnings else []
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="substrate")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an experiment file")
    run.add_argument("experiment")
    run.add_argument("--samples", type=int, help="override the ensemble size")
    run.add_argument("--out", help="directory to persist every intermediate system as JSON")
    cal = sub.add_parser("calibrate", help="fit the model engines to a reference surface (e.g. real quantum chemistry)")
    cal.add_argument("experiment", help="an experiment whose electronic-structure system produces the reference 2D surface")
    cal.add_argument("--out", help="write a ready-to-run calibrated experiment (YAML) here")
    cal.add_argument("--json", help="save the full calibration (parameters, covariance, diagnostics) here")
    cal.add_argument("--samples", type=int, default=100, help="ensemble size written into the calibrated experiment")
    cal.add_argument("--window", type=float, default=1.5, help="fit points up to this many eV above the surface minimum")
    cal.add_argument("--sigma", type=float, default=0.01, help="fit tolerance in eV, flat across the window")
    cal.add_argument("--no-validation", action="store_true", help="skip the leave-one-distance-out validation (faster)")
    tr = sub.add_parser("transfer", help="calibrate against several reference surfaces and test how far each calibration carries to the others")
    tr.add_argument("experiments", nargs="+", help="experiments whose electronic-structure systems each produce a reference 2D surface")
    tr.add_argument("--window", type=float, help="fit points up to this many eV above the surface minimum, for every reference "
                                                  "(default: the window each experiment file declares, else 1.5)")
    tr.add_argument("--sigma", type=float, default=0.01, help="fit tolerance in eV, flat across the window")
    tr.add_argument("--reference-distance", type=float,
                    help="where every fit defines its coupling, so the parameters share variables (default: the mean of the references' mid-range distances)")
    tr.add_argument("--json", help="save the cross-prediction numbers here")
    gui = sub.add_parser("gui", help="start the local web GUI (mission control: run, edit, watch the scales, explore the transfer study)")
    gui.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765)")
    gui.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    gui.add_argument("--root", help="the project folder that holds experiments/ (default: this checkout)")
    args = parser.parse_args(argv)

    if args.command == "gui":
        from .gui import serve
        return serve(args.port, args.root, open_browser=not args.no_browser)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "transfer":
        return _transfer(args)
    try:
        exp = load_experiment(args.experiment)
        n = exp.n_samples if args.samples is None else args.samples
        result = Pipeline(default_registry()).run(
            exp.system, exp.propagation, n_samples=n, seed=exp.seed, experiment_id=exp.id
        )
    except SubstrateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(report(result))
    if args.out:
        print(f"\nsaved to {result.save(args.out)}")
    return 0


def _calibrate(args) -> int:
    from .calibration import FitSettings, calibrate_evb_2d, calibrated_experiment, write_experiment
    try:
        exp = load_experiment(args.experiment)
        target = Pipeline(default_registry()).run(exp.system, [exp.system.scale]).final     # the reference surface
        calibration = calibrate_evb_2d(target, FitSettings(window_ev=args.window, sigma_ev=args.sigma),
                                       cross_validate=not args.no_validation)
        print(calibration.summary())
        if args.json:
            calibration.save(args.json)
            print(f"\ncalibration saved to {args.json}")
        if args.out:
            spec = calibrated_experiment(calibration, read_spec(args.experiment), n_samples=args.samples, seed=exp.seed)
            header = "\n".join([
                f"Calibrated model of '{target.name}', generated by:  python -m substrate calibrate {args.experiment}",
                "The 2D valence-bond model with parameters fitted to the reference surface, so it runs in milliseconds.",
                "The ensemble draws the fitted parameters JOINTLY from the fit covariance (they are strongly correlated).",
                "That spread is the fit's parameter uncertainty, NOT the model's error against the reference: see the",
                "calibration summary (per-distance barrier error, leave-one-out) and compare predictions directly.",
            ])
            print(f"calibrated experiment written to {write_experiment(spec, args.out, header)}")
    except SubstrateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _transfer(args) -> int:
    import json
    from pathlib import Path

    from .calibration import FitSettings
    from .transfer import fit_settings, format_report, make_reference, transfer_matrix
    try:
        pipeline = Pipeline(default_registry())
        solved = {}
        for path in args.experiments:
            exp = load_experiment(path)
            name = exp.id or Path(path).stem
            if name in solved:
                raise SubstrateError(f"two experiments are called '{name}': each reference needs its own id")
            solved[name] = (exp, pipeline.run(exp.system, [exp.system.scale]).final)
        mids = [0.5 * (float(np.min(s.obs("surface_r", "angstrom"))) + float(np.max(s.obs("surface_r", "angstrom"))))
                for _, s in solved.values() if "surface_r" in s.observables]
        r_ref = args.reference_distance if args.reference_distance is not None else (float(np.mean(mids)) if mids else None)
        base = FitSettings(window_ev=args.window or 1.5, sigma_ev=args.sigma, reference_distance=r_ref)
        references = {name: make_reference(name, exp.system, target, base if args.window else fit_settings(exp, base))
                      for name, (exp, target) in solved.items()}
        matrix = transfer_matrix(references, args.window)
        print(format_report(references, matrix))
        print(f"\n(every fit defines its coupling at {r_ref:.3f} angstrom; tolerance {args.sigma:g} eV; each reference's fit window is in the table above)")
        if args.json:
            out = {f"{s} -> {t}": {"shift_ev": p.shift_ev, "rmse_ev": p.rmse_ev, "max_error_ev": p.max_error_ev,
                                   "barrier_error_mean_abs_ev": p.barrier_error_mean_abs, "wells_wrong": p.wells_wrong}
                   for (s, t), p in matrix.items()}
            Path(args.json).write_text(json.dumps(out, indent=1))
            print(f"cross-prediction saved to {args.json}")
    except SubstrateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
