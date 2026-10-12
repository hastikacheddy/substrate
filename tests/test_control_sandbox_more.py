"""Cases of the step sandbox that the first tests did not pin, found by mutation testing: that the child really ends up with the scrubbed environment and nothing else, which names are
scrubbed by prefix alone, that a child which answers but will not exit is still stopped, how a memory error and a very long message arrive, and which classes may be rebuilt from a name."""
import io
import json
import os
import subprocess
import sys
import time

import pytest

from substrate import ResourceLimitError, ValidationError
from substrate.errors import SubstrateError
from substrate.qpu import QuantumPolicy, QuantumPolicyError
from substrate.control import sandbox
from substrate.control.sandbox import ProcessExecutor, StepTimeout, _child, _isolate_environment, _rehydrate, scrub_environment
from substrate.control.security import AuthorizationError, QuotaExceededError

from test_control_sandbox import request_for


# -- the environment ----------------------------------------------------------------------------------------------------------------
def test_names_that_are_credentials_only_by_their_prefix_are_scrubbed_too():
    env = {"IBM_BACKEND": "x", "QISKIT_SETTINGS": "x", "SUBSTRATE_S3_BUCKET": "b", "SUBSTRATE_DB_POOL": "5", "AWS_REGION": "eu", "PGHOST": "h", "MY_AUTHORITY": "a",
           "DATABASE_HOST": "h", "REDIS_HOST": "r", "MINIO_BUCKET": "m", "AZURE_REGION": "a", "GCP_ZONE": "z", "GOOGLE_PROJECT": "p", "PATH": "/bin", "TZ": "UTC"}
    assert scrub_environment(env) == {"PATH": "/bin", "TZ": "UTC"}


def test_the_keep_list_and_the_limit_variables_survive_even_when_their_names_look_like_credentials(monkeypatch):
    env = {"SUBSTRATE_MAX_TOKEN_COUNT": "7", "SUBSTRATE_max_session_draws": "9", "MY_TOKEN_PATH": "x", "SUBSTRATE_IMAGE_DIGEST": "sha256:abc", "SUBSTRATE_MAXIMUM_TOKEN": "y"}
    kept = scrub_environment(env)
    assert kept == {"SUBSTRATE_MAX_TOKEN_COUNT": "7", "SUBSTRATE_max_session_draws": "9", "SUBSTRATE_IMAGE_DIGEST": "sha256:abc"}      # a limit is policy; a similar-looking name is not
    monkeypatch.setattr(sandbox, "KEEP", sandbox.KEEP | {"MY_TOKEN_PATH"})
    assert scrub_environment(env)["MY_TOKEN_PATH"] == "x"                                      # and a name on the keep list is kept whatever it looks like


def test_a_process_that_isolates_its_environment_keeps_only_what_is_left_after_scrubbing():
    code = ("import os, sys, json\nsys.path.insert(0, %r)\nfrom substrate.control.sandbox import _isolate_environment\n_isolate_environment()\nprint(json.dumps(sorted(os.environ)))\n"
            % os.path.join(os.path.dirname(__file__), "..", "src"))
    env = {**os.environ, "MY_API_TOKEN": "hunter2", "AWS_ACCESS_KEY_ID": "AKIA", "SUBSTRATE_DB": "postgresql://u:p@h/d", "SUBSTRATE_MAX_ENSEMBLE": "5", "HARMLESS_SETTING": "v"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    left = json.loads(result.stdout)
    assert "MY_API_TOKEN" not in left and "AWS_ACCESS_KEY_ID" not in left and "SUBSTRATE_DB" not in left, result.stderr[-300:]
    assert "SUBSTRATE_MAX_ENSEMBLE" in left and "HARMLESS_SETTING" in left and "hunter2" not in result.stdout


class Pipe:
    """Stands in for the connection a child writes its answer to."""
    def __init__(self):
        self.sent, self.closed = [], False

    def send_bytes(self, data):
        self.sent.append(json.loads(data.decode()))

    def close(self):
        self.closed = True


def payload(**changes):
    _, request, _ = request_for()
    return json.dumps({"request": request.to_dict(), "memory_bytes": None, "cpu_seconds": None, "policy": None, **changes})


@pytest.fixture
def in_process(monkeypatch):
    """Run `_child` in this process without letting it clear this process's environment."""
    monkeypatch.setattr(sandbox, "_isolate_environment", lambda: None)


def test_the_child_isolates_its_environment_before_it_does_anything_else(monkeypatch):
    order = []
    monkeypatch.setattr(sandbox, "_isolate_environment", lambda: order.append("isolate"))
    monkeypatch.setattr(sandbox, "_apply_limits", lambda *a: order.append("limits"))
    monkeypatch.setattr(sandbox, "apply_step_data", lambda *a: order.append("step") or type("S", (), {"to_dict": lambda self: {}})())
    pipe = Pipe()
    _child(pipe, payload())
    assert order == ["isolate", "limits", "step"] and pipe.sent == [{"status": "ok", "system": {}}] and pipe.closed


def test_a_memory_error_in_the_child_arrives_as_a_resource_limit(in_process, monkeypatch):
    def out_of_memory(*a):
        raise MemoryError()
    monkeypatch.setattr(sandbox, "apply_step_data", out_of_memory)
    pipe = Pipe()
    _child(pipe, payload())
    assert pipe.sent == [{"status": "error", "type": "ResourceLimitError", "message": "the step exceeded its memory limit"}] and pipe.closed
    assert isinstance(_rehydrate("ResourceLimitError", pipe.sent[0]["message"]), ResourceLimitError)


def test_an_error_message_longer_than_2000_characters_is_cut_before_it_is_sent(in_process, monkeypatch):
    def refuse(*a):
        raise ValidationError("x" * 5000)
    monkeypatch.setattr(sandbox, "apply_step_data", refuse)
    pipe = Pipe()
    _child(pipe, payload())
    assert pipe.sent[0]["type"] == "ValidationError" and len(pipe.sent[0]["message"]) == 2000


def test_the_quantum_policy_the_worker_holds_is_the_one_the_child_enforces(in_process, monkeypatch):
    seen = []
    monkeypatch.setattr(sandbox, "apply_step_data", lambda pipeline, request, policy: seen.append(policy) or type("S", (), {"to_dict": lambda self: {}})())
    policy = QuantumPolicy(max_qubits=3, max_shots=7, providers=("simulator", "lab-device"), hardware_providers=("lab-device",))
    from dataclasses import asdict
    _child(Pipe(), payload(policy=asdict(policy)))
    _child(Pipe(), payload(policy=None))
    assert seen == [policy, None]                                                              # lists from JSON come back as the tuples they were


# -- what comes back across the boundary --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind, cls", [("ValidationError", ValidationError), ("ResourceLimitError", ResourceLimitError), ("QuantumPolicyError", QuantumPolicyError),
                                       ("QuotaExceededError", QuotaExceededError), ("AuthorizationError", AuthorizationError)])
def test_the_errors_the_platform_raises_keep_their_class_across_the_process_boundary(kind, cls):
    error = _rehydrate(kind, "why")
    assert type(error) is cls and str(error) == "why" and isinstance(error, SubstrateError)


@pytest.mark.parametrize("kind", ["Circuit", "Gate", "QuantumPolicy", "Role", "Action", "Principal", "StatevectorSimulator", "Path", "dict", "object", "Exception"])
def test_a_class_that_merely_shares_a_name_with_something_in_those_modules_is_not_instantiated(kind):
    error = _rehydrate(kind, "x")
    assert type(error) is RuntimeError and kind in str(error)


# -- a child that answers or overruns ------------------------------------------------------------------------------------------------------
def answer_then_linger(connection, payload):
    connection.send_bytes(json.dumps({"status": "ok", "system": {"done": True}}).encode())
    connection.close()
    time.sleep(120)


def sleep_a_long_time(connection, payload):
    time.sleep(120)


def test_a_child_that_answers_but_will_not_exit_is_stopped_once_its_answer_is_in():
    pipeline, request, _ = request_for()
    executor = ProcessExecutor(target=answer_then_linger, memory_bytes=None)
    assert executor.apply(pipeline, request) == {"done": True}
    assert not executor.last_process.is_alive()


class TimedProcess:
    """Wraps a child process and records how long each `join` really waited."""

    def __init__(self, process):
        self.process, self.joins = process, []

    def __getattr__(self, name):
        return getattr(self.process, name)

    def join(self, timeout=None):
        started = time.perf_counter()
        self.process.join(timeout)
        self.joins.append(time.perf_counter() - started)


def test_a_step_that_overruns_its_deadline_is_stopped_at_once_not_after_a_grace_period():
    """Measured on the waits after the deadline, not on the whole call: starting a process can take seconds on a busy machine, and that is not what is being tested. Without the
    explicit kill the cleanup would wait the full five seconds of `join(5)` for a child that sleeps for two minutes."""
    pipeline, request, _ = request_for(timeout_s=1.0)
    executor = ProcessExecutor(target=sleep_a_long_time, memory_bytes=None)
    real, made = executor.context, []

    class Context:
        Pipe = staticmethod(real.Pipe)

        @staticmethod
        def Process(**kwargs):
            made.append(TimedProcess(real.Process(**kwargs)))
            return made[-1]

    executor.context = Context
    with pytest.raises(StepTimeout):
        executor.apply(pipeline, request)
    (process,) = made
    assert process.joins and max(process.joins) < 3.5, process.joins
    assert not process.is_alive()


# -- limits (POSIX only: they run on Linux, in CI) ---------------------------------------------------------------------------------------------
def run_python(code, timeout=120):
    prefix = "import sys\nsys.path.insert(0, %r)\nfrom substrate.control.sandbox import _apply_limits\nimport resource\n" % os.path.join(os.path.dirname(__file__), "..", "src")
    return subprocess.run([sys.executable, "-c", prefix + code], capture_output=True, text=True, timeout=timeout)


def test_by_default_a_step_may_not_use_more_than_four_gibibytes_of_memory():
    assert ProcessExecutor().memory_bytes == 4 * 1024**3 and ProcessExecutor().cpu_seconds is None


@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX only")
def test_a_child_cannot_write_a_file_larger_than_ten_mebibytes(tmp_path):
    result = run_python("_apply_limits(None, None)\n"
                        "print(resource.getrlimit(resource.RLIMIT_FSIZE)[0])\n"
                        "try:\n    open(%r, 'wb').write(b'x' * (11 * 1024 * 1024))\n    print('wrote')\nexcept OSError as error:\n    print('refused', error.errno)\n" % str(tmp_path / "big.bin"))
    lines = result.stdout.split()
    assert lines[0] == str(10 * 1024 * 1024) and lines[1] == "refused", result.stderr[-300:]


@pytest.mark.skipif(sys.platform == "win32", reason="resource limits are POSIX only")
def test_a_limit_is_never_set_above_the_hard_limit_the_process_already_has():
    result = run_python("resource.setrlimit(resource.RLIMIT_CPU, (50, 100))\n_apply_limits(None, 10**9)\nprint(resource.getrlimit(resource.RLIMIT_CPU))\n"
                        "resource.setrlimit(resource.RLIMIT_AS, (2**40, 2**41))\n_apply_limits(2**50, None)\nprint(resource.getrlimit(resource.RLIMIT_AS))\n")
    assert result.stdout.split("\n")[:2] == ["(100, 100)", "(%d, %d)" % (2**41, 2**41)], result.stderr[-300:]
