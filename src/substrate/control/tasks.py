"""The work a task does, and how an experiment becomes a graph of tasks.

An experiment's workflow is the same shape every time:

    validate -> ingest -> (one task per planned scale step, in order) -> [uncertainty] -> check -> report

* `validate` parses the experiment (every limit and range check applies here) and plans the route between scales; its output is the root system.
* `ingest` confirms the datasets the experiment is pinned to exist, unchanged, at the versions it names.
* each step task applies one engine or translator to the previous task's system. Steps that need a quantum-chemistry program go to the `chemistry` queue.
* `uncertainty` re-runs the chain with the root's uncertain parameters drawn at random, if an ensemble was asked for.
* `check` looks at the final observables (finite? any warnings from the models?) and `report` writes the result, the assumptions it rests on and a digest.

Handlers return `(output, lineage)`: the output is stored as a JSON artifact and becomes the task's `output`; the lineage, if any, is appended to the run's hash-chained
lineage log by the worker only after the task has been accepted, so a result that was discarded leaves no trace in it.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..engines.qc_scan import QCScanEngine
from ..errors import DatasetUnavailableError, ValidationError
from ..experiment import parse_experiment
from ..ir import Scale, ScientificSystem
from ..limits import require
from ..pipeline import Pipeline, Step
from .artifacts import Artifacts
from .registry import Registry, TaskSpec, sha256_json
from .repro import observables_summary, result_digest
from .steps import StepRequest
from .taskqueue import ClaimedTask

DEFAULT_TIMEOUT_S = 600.0
CHEMISTRY_TIMEOUT_S = 6 * 3600.0
STEP_KINDS = ("solve", "translate")


def queue_for(step: Step, backend: dict | None = None) -> str:
    """Where a step runs: chemistry if it needs a quantum-chemistry program, quantum if the experiment asked for the quantum solver, else cpu."""
    if isinstance(step.component, QCScanEngine):
        return "chemistry"
    return "quantum" if (backend or {}).get("type") == "quantum" else "cpu"


def check_dataset_pins(pins) -> list[dict]:
    if not isinstance(pins, list) or any(not isinstance(p, dict) or set(p) != {"id", "version"} or not isinstance(p["id"], str)
                                         or isinstance(p["version"], bool) or not isinstance(p["version"], int) or p["version"] < 1 for p in pins):
        raise ValidationError("datasets must be a list of {id, version} with a whole version number of at least 1")
    return [{"id": p["id"], "version": p["version"]} for p in pins]


def is_transfer_spec(spec) -> bool:
    return isinstance(spec, dict) and "transfer" in spec


def parse_transfer_spec(spec: dict) -> dict:
    """The settings of a transfer study: which experiments are the references, the fit window (None: each reference's own) and the fit tolerance."""
    if set(spec) != {"transfer"} or not isinstance(spec["transfer"], dict):
        raise ValidationError("a transfer study is {transfer: {references: [...], window_ev: ..., sigma_ev: ...}} and nothing else")
    body = spec["transfer"]
    unknown = set(body) - {"references", "window_ev", "sigma_ev"}
    references = body.get("references")
    if unknown or not isinstance(references, list) or not all(isinstance(r, str) for r in references):
        raise ValidationError(f"a transfer study takes only references (a list of experiment ids), window_ev and sigma_ev (got {sorted(body)})")
    if len(set(references)) != len(references):
        raise ValidationError("a transfer study lists the same reference twice")
    if len(references) < 2:
        raise ValidationError("a transfer study needs at least two references")
    require("max_references", len(references), "references in one transfer study")
    window, sigma = body.get("window_ev"), body.get("sigma_ev", 0.05)
    for name, value, allow_none in (("window_ev", window, True), ("sigma_ev", sigma, False)):
        if value is None and allow_none:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 50:
            raise ValidationError(f"{name} must be a number above 0 and at most 50 (got {value!r})")
    return {"references": list(references), "window_ev": None if window is None else float(window), "sigma_ev": float(sigma)}


def build_transfer_tasks(spec: dict, resolve, pipeline: Pipeline) -> list[TaskSpec]:
    """The task graph of a transfer study. `resolve(id)` returns the stored spec of a reference experiment of the caller's tenant, or None."""
    study = parse_transfer_spec(spec)
    tasks = [TaskSpec("validate", "validate_transfer", payload={})]
    for reference in study["references"]:
        reference_spec = resolve(reference)
        if reference_spec is None:
            raise ValidationError(f"no experiment {reference} to use as a reference")
        if is_transfer_spec(reference_spec):
            raise ValidationError(f"{reference} is itself a transfer study, not a reference surface")
        experiment = parse_experiment(reference_spec)
        first = pipeline.plan(experiment.system, [experiment.system.scale])[0]
        queue = queue_for(first)
        tasks.append(TaskSpec(f"reference:{reference}", "reference", queue=queue, payload={"experiment": reference}, depends_on=["validate"],
                              max_attempts=2 if queue == "chemistry" else 3, timeout_s=CHEMISTRY_TIMEOUT_S if queue == "chemistry" else DEFAULT_TIMEOUT_S))
    tasks.append(TaskSpec("transfer", "transfer", depends_on=[t.name for t in tasks[1:]], timeout_s=DEFAULT_TIMEOUT_S * 6,
                          payload={"window_ev": study["window_ev"], "sigma_ev": study["sigma_ev"]}))
    tasks.append(TaskSpec("report", "transfer_report", depends_on=["transfer"]))
    return tasks


def build_tasks(spec: dict, pipeline: Pipeline, *, n_samples: int, datasets: list[dict] | None = None) -> list[TaskSpec]:
    """The task graph for an experiment. Raises for a spec that cannot be run (bad numbers, no route between the scales, an ensemble with nothing to vary)."""
    experiment = parse_experiment(spec)
    if n_samples > 0 and not pipeline._uncertain_names(experiment.system):
        raise ValidationError("an ensemble was requested but no parameter has a sigma to vary")
    steps = pipeline.plan(experiment.system, experiment.propagation)
    pins = check_dataset_pins(datasets or [])
    backend = experiment.backend
    tasks = [TaskSpec("validate", "validate"), TaskSpec("ingest", "ingest", payload={"datasets": pins}, depends_on=["validate"])]
    previous = "ingest"
    names = []
    for i, step in enumerate(steps):
        queue = queue_for(step, backend)
        name = f"{i:02d}-{step.label}"
        tasks.append(TaskSpec(name, step.kind, queue=queue, payload={"index": i, "label": step.label}, depends_on=[previous],
                              max_attempts=2 if queue == "chemistry" else 3, timeout_s=CHEMISTRY_TIMEOUT_S if queue == "chemistry" else DEFAULT_TIMEOUT_S,
                              resources={"cpu": 1}))
        previous = name
        names.append(name)
    if n_samples > 0:
        tasks.append(TaskSpec("uncertainty", "uncertainty", depends_on=[previous], timeout_s=DEFAULT_TIMEOUT_S * 6, resources={"cpu": 1}))
        previous = "uncertainty"
    tasks.append(TaskSpec("check", "check", depends_on=[previous]))
    tasks.append(TaskSpec("report", "report", depends_on=["check"]))
    return tasks


# -- the context a handler gets --------------------------------------------------------------------------------------------------------------
class TaskContext:
    def __init__(self, task: ClaimedTask, registry: Registry, artifacts: Artifacts, pipeline: Pipeline, executor):
        self.task, self.registry, self.artifacts, self.pipeline, self.executor = task, registry, artifacts, pipeline, executor
        self.run = registry.run_in_tenant(task.tenant_id, task.run_id)
        self.experiment = registry.experiment_in_tenant(task.tenant_id, self.run["experiment_id"])
        self._rows: list[dict] | None = None
        self._cache: dict[str, dict] = {}

    def rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = self.registry.tasks_in_tenant(self.task.tenant_id, self.task.run_id)
        return self._rows

    def _output(self, row: dict) -> dict:
        if row["state"] != "succeeded" or not row["output"]:
            raise LookupError(f"task {row['name']} has no output (it is {row['state']})")
        if row["id"] not in self._cache:
            self._cache[row["id"]] = self.artifacts.read_json(self.task.tenant_id, row["output"])
        return self._cache[row["id"]]

    def output_of(self, name: str) -> dict:
        for row in self.rows():
            if row["name"] == name:
                return self._output(row)
        raise LookupError(f"this run has no task '{name}'")

    def step_output(self, index: int) -> dict:
        for row in self.rows():
            if row["kind"] in STEP_KINDS and row["payload"].get("index") == index:
                return self._output(row)
        raise LookupError(f"this run has no step {index}")

    def final_system(self) -> ScientificSystem:
        if self.run["n_samples"]:
            return ScientificSystem.from_dict(self.output_of("uncertainty")["final"])
        steps = self.output_of("validate")["steps"]
        return ScientificSystem.from_dict(self.step_output(len(steps) - 1)["system"])


Handler = Callable[[TaskContext], "tuple[dict, dict | None]"]


def h_validate(ctx: TaskContext):
    experiment = parse_experiment(ctx.experiment["spec"])
    steps = ctx.pipeline.plan(experiment.system, experiment.propagation)
    return {"root": experiment.system.to_dict(), "propagation": [s.label for s in experiment.propagation], "steps": [s.label for s in steps],
            "root_fingerprint": experiment.system.fingerprint()}, None


def h_ingest(ctx: TaskContext):
    found = []
    for pin in ctx.task.payload["datasets"]:
        row = ctx.registry.get_dataset(pin["id"], pin["version"])
        if row is None:
            raise DatasetUnavailableError(f"dataset {pin['id']} version {pin['version']} is not in the registry")
        if sha256_json(row["records"]) != row["checksum"]:
            raise ValidationError(f"dataset {pin['id']} version {pin['version']} no longer matches its recorded checksum")
        found.append({"id": pin["id"], "version": pin["version"], "checksum": row["checksum"], "source": row["source"], "license": row["license"]})
    return {"datasets": found}, None


def h_step(ctx: TaskContext):
    validate = ctx.output_of("validate")
    root_data, propagation = validate["root"], validate["propagation"]
    index, label = ctx.task.payload["index"], ctx.task.payload["label"]
    if validate["steps"][index] != label:
        raise ValidationError(f"the planned route no longer matches the task (step {index} is '{validate['steps'][index]}', the task expects '{label}')")
    system_data = root_data if index == 0 else ctx.step_output(index - 1)["system"]
    started = time.perf_counter()
    backend = parse_experiment(ctx.experiment["spec"]).backend
    out_data = ctx.executor.apply(ctx.pipeline, StepRequest(root_data, propagation, index, system_data, ctx.task.timeout_s, backend))
    before, after = ScientificSystem.from_dict(system_data).fingerprint(), ScientificSystem.from_dict(out_data).fingerprint()
    lineage = {"task": ctx.task.id, "index": index, "label": label, "input_fingerprint": before, "output_fingerprint": after}
    return {"system": out_data, "index": index, "label": label, "input_fingerprint": before, "output_fingerprint": after,
            "seconds": time.perf_counter() - started}, lineage


def h_uncertainty(ctx: TaskContext):
    validate = ctx.output_of("validate")
    root = ScientificSystem.from_dict(validate["root"])
    steps = ctx.pipeline.plan(root, validate["propagation"])
    nominal = [ScientificSystem.from_dict(ctx.step_output(i)["system"]) for i in range(len(steps))]
    trace, info = ctx.pipeline._with_ensemble(root, steps, nominal, ctx.run["n_samples"], ctx.run["seed"])
    return {"final": trace[-1].to_dict(), "ensemble": info}, {"task": ctx.task.id, "label": "ensemble", "n_requested": info["n_requested"], "n_ok": info["n_ok"]}


def h_check(ctx: TaskContext):
    final = ctx.final_system()
    if not final.observables:
        raise ValidationError("the final system has no observables")
    non_finite = [n for n, q in final.observables.items() if not np.all(np.isfinite(np.asarray(q.value, dtype=float)))]
    warnings = []
    for record in final.provenance:
        for w in record.warnings:
            text = f"[{record.name}] {w}"
            if text not in warnings:
                warnings.append(text)
    return {"non_finite": non_finite, "warnings": warnings, "n_observables": len(final.observables)}, None


def h_report(ctx: TaskContext):
    tenant, run = ctx.task.tenant_id, ctx.run
    final = ctx.final_system()
    summary = observables_summary(final.observables)
    digest = result_digest(summary)
    check = ctx.output_of("check")
    ingest = ctx.output_of("ingest")
    records = [r.to_dict() for r in final.provenance]
    assumptions = sorted({a for r in records for a in r["approximations"]})
    discarded = [{"hop": r["name"], "scale_in": r["scale_in"], "scale_out": r["scale_out"], "information": r["discarded"]} for r in records if r.get("discarded")]
    scalars = {n: s["values"][0] for n, s in summary.items() if not s["shape"] and s["values"]}
    report = {
        "run": run["id"], "experiment": run["experiment_id"], "name": ctx.experiment["name"], "seed": run["seed"], "n_samples": run["n_samples"],
        "environment_hash": run["environment_hash"], "spec_sha256": ctx.experiment["spec_sha256"], "datasets": ingest["datasets"],
        "backend": parse_experiment(ctx.experiment["spec"]).backend,
        "steps": [{**{k: r[k] for k in ("step", "name", "version", "scale_in", "scale_out", "backend", "approximations", "warnings", "seconds")},
                   "discarded": r.get("discarded", []), "validity": "valid with warnings" if r["warnings"] else "valid"} for r in records],
        "assumptions": assumptions, "discarded_information": discarded, "warnings": check["warnings"], "non_finite": check["non_finite"], "scalar_observables": scalars,
        "observables": {n: {"unit": s["unit"], "shape": s["shape"]} for n, s in summary.items()}, "result_digest": digest,
    }
    observables = ctx.artifacts.write_json(tenant, summary, name="final_observables.json", run_id=run["id"])
    report_ref = ctx.artifacts.write_json(tenant, report, name="report.json", run_id=run["id"])
    markdown = ctx.artifacts.write(tenant, render_markdown(report).encode("utf-8"), name="report.md", media_type="text/markdown", run_id=run["id"])
    return {"observables": observables.sha256, "report": report_ref.sha256, "markdown": markdown.sha256, "result_digest": digest}, None


def render_markdown(report: dict) -> str:
    lines = [f"# {report['name']}", "", f"Run {report['run']} of {report['experiment']}; seed {report['seed']}; {report['n_samples']} ensemble draws.",
             f"Environment {report['environment_hash'][:12]}; result digest {report['result_digest'][:12]}.", "", "## Route", ""]
    lines += [f"{i + 1}. {s['step']} {s['name']} ({s['scale_in']} -> {s['scale_out']})" for i, s in enumerate(report["steps"])]
    lines += ["", "## Assumptions made along the way", ""] + ([f"- {a}" for a in report["assumptions"]] or ["- none recorded"])
    lines += ["", "## Information discarded between scales", ""]
    lines += [f"- {hop['hop']} ({hop['scale_in']} -> {hop['scale_out']}): " + "; ".join(hop["information"]) for hop in report["discarded_information"]] or ["- none recorded"]
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in report["warnings"]] or ["- none"])
    if report["non_finite"]:
        lines += ["", f"Observables with non-finite values: {', '.join(report['non_finite'])}"]
    lines += ["", "## Datasets", ""] + ([f"- {d['id']} v{d['version']} ({d['source']}; {d['license']})" for d in report["datasets"]] or ["- none"])
    lines += ["", "## Scalar observables", ""] + [f"- {n} = {v:.6g} {report['observables'][n]['unit']}" for n, v in sorted(report["scalar_observables"].items())]
    return "\n".join(lines) + "\n"


# -- transfer studies ------------------------------------------------------------------------------------------------------------------------------
def _study(ctx: TaskContext) -> dict:
    return parse_transfer_spec(ctx.experiment["spec"])


def h_validate_transfer(ctx: TaskContext):
    study = _study(ctx)
    found = []
    for reference in study["references"]:
        row = ctx.registry.experiment_in_tenant(ctx.task.tenant_id, reference)
        if row is None:
            raise ValidationError(f"no experiment {reference} to use as a reference")
        parse_experiment(row["spec"])
        found.append({"id": reference, "name": row["name"], "spec_sha256": row["spec_sha256"]})
    return {"references": found, "window_ev": study["window_ev"], "sigma_ev": study["sigma_ev"]}, None


def h_reference(ctx: TaskContext):
    reference = ctx.task.payload["experiment"]
    row = ctx.registry.experiment_in_tenant(ctx.task.tenant_id, reference)
    if row is None:
        raise ValidationError(f"no experiment {reference} to use as a reference")
    experiment = parse_experiment(row["spec"])
    solved = ctx.pipeline.run(experiment.system, [experiment.system.scale], experiment_id=reference).final
    return {"experiment": reference, "name": row["name"], "solved": solved.to_dict()}, {"task": ctx.task.id, "label": f"reference {reference}",
                                                                                        "output_fingerprint": solved.fingerprint()}


def h_transfer(ctx: TaskContext):
    from dataclasses import replace

    from ..calibration import FitSettings
    from ..transfer import fit_settings, format_report, make_reference, parameter_table, transfer_matrix

    study = _study(ctx)
    window, sigma = ctx.task.payload["window_ev"], ctx.task.payload["sigma_ev"]
    loaded = []
    for reference in study["references"]:
        out = ctx.output_of(f"reference:{reference}")
        experiment = parse_experiment(ctx.registry.experiment_in_tenant(ctx.task.tenant_id, reference)["spec"])
        loaded.append((reference, experiment, ScientificSystem.from_dict(out["solved"])))
    mids = [0.5 * (float(np.min(s.obs("surface_r", "angstrom"))) + float(np.max(s.obs("surface_r", "angstrom")))) for _, _, s in loaded if "surface_r" in s.observables]
    r_ref = float(np.mean(mids)) if mids else None
    base = FitSettings(window_ev=window or 1.5, sigma_ev=sigma, reference_distance=r_ref)
    references = {}
    for name, experiment, solved in loaded:
        settings = fit_settings(experiment, base)
        references[name] = make_reference(name, experiment.system, solved, replace(settings, window_ev=window) if window else settings)
    matrix = transfer_matrix(references, window)
    names = list(references)
    pairs = {f"{s} -> {t}": {"window_ev": p.window_ev, "shift_ev": p.shift_ev, "rmse_window_ev": p.rmse_window, "max_error_window_ev": p.max_error_ev[f"{p.window_ev:g}"],
                             "rmse_ev": p.rmse_ev, "max_error_ev": p.max_error_ev, "barrier_error_mean_abs_ev": p.barrier_error_mean_abs,
                             "wells_wrong": p.wells_wrong} for (s, t), p in matrix.items()}
    parameters = {n: {k: (None if v is None else list(v)) for k, v in row.items()} for n, row in parameter_table(references).items()}
    return {"names": names, "pairs": pairs, "parameters": parameters, "reference_distance": r_ref, "text": format_report(references, matrix)}, \
        {"task": ctx.task.id, "label": "transfer", "references": names}


METRICS = (("shift_ev", "eV"), ("rmse_window_ev", "eV"), ("max_error_window_ev", "eV"), ("barrier_error_mean_abs_ev", "eV"), ("wells_wrong", "1"))


def transfer_summary(result: dict) -> dict:
    """The transfer result as observables: one n-by-n matrix per metric (row: source, column: target; NaN where a pair was not computed)."""
    names = result["names"]
    summary = {}
    for metric, unit in METRICS:
        values = []
        for source in names:
            for target in names:
                pair = result["pairs"].get(f"{source} -> {target}")
                value = None if pair is None else pair[metric]
                values.append(float("nan") if value is None else float(value))
        summary[metric] = {"unit": unit, "shape": [len(names), len(names)], "values": values}
    return summary


def h_transfer_report(ctx: TaskContext):
    tenant, run = ctx.task.tenant_id, ctx.run
    result = ctx.output_of("transfer")
    summary = transfer_summary(result)
    digest = result_digest(summary)
    observables = ctx.artifacts.write_json(tenant, summary, name="final_observables.json", run_id=run["id"])
    report = {"run": run["id"], "experiment": run["experiment_id"], "name": ctx.experiment["name"], "references": result["names"],
              "reference_distance": result["reference_distance"], "environment_hash": run["environment_hash"], "result_digest": digest}
    report_ref = ctx.artifacts.write_json(tenant, report, name="report.json", run_id=run["id"])
    markdown = ctx.artifacts.write(tenant, ("# " + ctx.experiment["name"] + "\n\n```\n" + result["text"] + "\n```\n").encode("utf-8"), name="report.md",
                                   media_type="text/markdown", run_id=run["id"])
    return {"observables": observables.sha256, "report": report_ref.sha256, "markdown": markdown.sha256, "result_digest": digest}, None


HANDLERS: dict[str, Handler] = {
    "validate": h_validate, "ingest": h_ingest, "solve": h_step, "translate": h_step, "uncertainty": h_uncertainty, "check": h_check, "report": h_report,
    "validate_transfer": h_validate_transfer, "reference": h_reference, "transfer": h_transfer, "transfer_report": h_transfer_report,
}
