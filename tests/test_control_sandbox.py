"""Running a step in its own process: same answers as in-process, failures that keep their meaning, a deadline that holds, and nothing unpickled from the child."""
import json
import os
import sys

import pytest

from substrate import ResourceLimitError, ValidationError, default_registry
from substrate.pipeline import Pipeline
from substrate.control.sandbox import (ProcessExecutor, StepCrashed, StepTimeout, _apply_limits, _rehydrate, scrub_environment)
from substrate.control.steps import InlineExecutor, StepRequest, apply_step_data
from substrate.experiment import parse_experiment, read_spec
from substrate.ir import ScientificSystem

from conftest import EXPERIMENTS


def request_for(name="proton_transfer", index=0, system=None, timeout_s=120.0, edit=None):
    spec = read_spec(EXPERIMENTS / f"{name}.yaml")
    if edit:
        edit(spec)
    experiment = parse_experiment(spec)
    pipeline = Pipeline(default_registry())
    steps = pipeline.plan(experiment.system, experiment.propagation)
    root = experiment.system.to_dict()
    return pipeline, StepRequest(root, [s.label for s in experiment.propagation], index, system if system is not None else root, timeout_s), len(steps)


# -- the environment ----------------------------------------------------------------------------------------------------------------
def test_anything_that_looks_like_a_credential_is_removed_and_ordinary_variables_stay():
    env = {"PATH": "/bin", "HOME": "/home/u", "SYSTEMROOT": "C:\\Windows", "LANG": "C", "OMP_NUM_THREADS": "1", "SUBSTRATE_MAX_ENSEMBLE": "50",
           "SUBSTRATE_IMAGE_DIGEST": "sha256:abc", "DATABASE_URL": "postgres://u:p@h/db", "AWS_ACCESS_KEY_ID": "AKIA", "AWS_REGION": "eu",
           "IBM_QUANTUM_TOKEN": "t", "MY_API_KEY": "k", "github_token": "g", "DB_PASSWORD": "p", "SESSION_COOKIE": "c", "OAUTH_CLIENT_SECRET": "s",
           "REDIS_URL": "redis://", "MINIO_ROOT_USER": "u", "PGHOST": "h", "SUBSTRATE_DB_URL": "sqlite://", "GOOGLE_APPLICATION_CREDENTIALS": "/k.json",
           "EDITOR": "vim", "TZ": "UTC"}
    clean = scrub_environment(env)
    assert set(clean) == {"PATH", "HOME", "SYSTEMROOT", "LANG", "OMP_NUM_THREADS", "SUBSTRATE_MAX_ENSEMBLE", "SUBSTRATE_IMAGE_DIGEST", "EDITOR", "TZ"}
    assert clean["SUBSTRATE_MAX_ENSEMBLE"] == "50"                                          # the limits must reach the step: they are policy, not secrets


def test_scrubbing_does_not_change_the_dict_it_is_given():
    env = {"MY_SECRET": "x", "PATH": "p"}
    scrub_environment(env)
    assert env == {"MY_SECRET": "x", "PATH": "p"}


# -- running a step ----------------------------------------------------------------------------------------------------------------------
def test_a_step_in_its_own_process_gives_the_same_system_as_in_this_one():
    pipeline, request, _ = request_for()
    inline = InlineExecutor().apply(pipeline, request)
    isolated = ProcessExecutor(memory_bytes=None).apply(pipeline, request)
    assert ScientificSystem.from_dict(isolated).fingerprint() == ScientificSystem.from_dict(inline).fingerprint()
    assert isolated["observables"].keys() == inline["observables"].keys() and isolated["provenance"][-1]["name"] == inline["provenance"][-1]["name"]


def test_the_childs_answer_can_feed_the_next_step():
    pipeline, first, total = request_for()
    out = ProcessExecutor(memory_bytes=None).apply(pipeline, first)
    _, second, _ = request_for(index=1, system=out)
    nxt = ProcessExecutor(memory_bytes=None).apply(pipeline, second)
    assert ScientificSystem.from_dict(nxt).scale != ScientificSystem.from_dict(out).scale and total == 3


def test_a_refusal_in_the_child_arrives_as_the_same_kind_of_error():
    def edit(spec):
        (spec["experiment"] if "experiment" in spec else spec)["system"]["parameters"]["oo_equilibrium"]["value"] = 2.3

    pipeline, request, _ = request_for("proton_transfer_molecular", index=0, edit=edit)
    system = request.system
    for index in (0, 1):                                                                  # the electronic solve and the translation run fine; the molecular solve refuses
        system = apply_step_data(pipeline, request_for("proton_transfer_molecular", index=index, system=system, edit=edit)[1]).to_dict()
    _, request, _ = request_for("proton_transfer_molecular", index=2, system=system, edit=edit)
    with pytest.raises(ValidationError, match="fewer than two wells"):
        ProcessExecutor(memory_bytes=None).apply(pipeline, request)


def test_the_childs_limits_are_the_same_as_the_workers(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MAX_GRID_POINTS", "1000")

    def edit(spec):
        (spec["experiment"] if "experiment" in spec else spec)["system"]["parameters"]["n_scan"] = {"value": 5000, "unit": "1"}

    pipeline, request, _ = request_for("proton_transfer_electronic", index=0, edit=edit)
    with pytest.raises(ResourceLimitError, match="n_scan: 5,000 exceeds the limit of 1,000"):
        ProcessExecutor(memory_bytes=None).apply(pipeline, request)


# -- a child that misbehaves ------------------------------------------------------------------------------------------------------------
def crash(connection, payload):                          # a module-level function, so that it can be started in a fresh interpreter
    os._exit(3)


def say_garbage(connection, payload):
    connection.send_bytes(b"\xff\xfe not json at all")
    connection.close()


def say_wrong_shape(connection, payload):
    connection.send_bytes(json.dumps(["not", "a", "message"]).encode())
    connection.close()


def send_a_pickle(connection, payload):
    import pickle

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > pwned.txt",))

    connection.send_bytes(pickle.dumps(Evil()))
    connection.close()


def sleep_forever(connection, payload):
    import time
    time.sleep(600)


def claim_a_custom_error(connection, payload):
    connection.send_bytes(json.dumps({"status": "error", "type": "SystemExit", "message": "exit"}).encode())
    connection.close()


def test_a_child_that_dies_without_answering_is_a_crash_not_a_hang():
    pipeline, request, _ = request_for()
    with pytest.raises(StepCrashed, match="exit code 3"):
        ProcessExecutor(target=crash, memory_bytes=None).apply(pipeline, request)


@pytest.mark.parametrize("target", [say_garbage, say_wrong_shape])
def test_an_answer_that_is_not_a_result_is_refused(target):
    pipeline, request, _ = request_for()
    with pytest.raises(StepCrashed, match="not a result"):
        ProcessExecutor(target=target, memory_bytes=None).apply(pipeline, request)


def test_a_pickle_sent_by_the_child_is_never_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline, request, _ = request_for()
    with pytest.raises(StepCrashed, match="not a result"):
        ProcessExecutor(target=send_a_pickle, memory_bytes=None).apply(pipeline, request)
    assert not (tmp_path / "pwned.txt").exists()                                             # `__reduce__` never ran, here or in the child's name


def test_a_step_that_overruns_its_deadline_is_killed():
    pipeline, request, _ = request_for(timeout_s=1.5)
    executor = ProcessExecutor(target=sleep_forever, memory_bytes=None)
    with pytest.raises(StepTimeout, match="did not finish within 1.5 s"):
        executor.apply(pipeline, request)
    executor.last_process.join(10)
    assert not executor.last_process.is_alive() and isinstance(StepTimeout("x"), TimeoutError)


def test_an_error_type_the_platform_does_not_know_does_not_become_a_substrate_error():
    error = _rehydrate("SystemExit", "exit")
    assert type(error) is RuntimeError and "SystemExit" in str(error)
    assert isinstance(_rehydrate("ValidationError", "bad"), ValidationError) and isinstance(_rehydrate("ResourceLimitError", "big"), ResourceLimitError)
    assert type(_rehydrate("NoSuchError", "x")) is RuntimeError
    assert type(_rehydrate("open", "x")) is RuntimeError                                    # a builtin name is not looked up as an error class


# -- limits (POSIX only) ------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX only; on Windows the deadline and the environment scrub apply")
def test_a_child_cannot_allocate_past_its_memory_limit():
    import subprocess
    code = ("import sys\nsys.path.insert(0, %r)\nfrom substrate.control.sandbox import _apply_limits\n_apply_limits(400 * 1024 * 1024, None)\n"
            "try:\n    bytearray(1024 * 1024 * 1024)\nexcept MemoryError:\n    print('memory-error')\n") % os.path.join(os.path.dirname(__file__), "..", "src")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.stdout.strip() == "memory-error", result.stderr[-300:]


@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX only")
def test_the_limits_never_exceed_what_the_system_already_allows():
    import resource
    before = resource.getrlimit(resource.RLIMIT_CPU)
    try:
        _apply_limits(None, 10**9)
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        assert hard == before[1] and (hard == resource.RLIM_INFINITY or soft <= hard)
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, before)


def test_limits_are_a_no_op_without_the_resource_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)                                       # as on Windows: `import resource` fails
    _apply_limits(10**9, 10)                                                                  # must not raise
