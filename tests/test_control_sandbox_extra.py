"""Two more cases of the sandbox found by mutation testing: an error class that cannot be rebuilt from a bare message, and the default for how a worker runs its steps."""
import pytest

from substrate.errors import SubstrateError
from substrate import errors
from substrate.control.sandbox import _rehydrate, default_isolation


def test_an_error_class_that_cannot_be_built_from_a_message_still_arrives_as_a_platform_error(monkeypatch):
    class NeedsTwoArguments(SubstrateError):
        def __init__(self, first, second):
            super().__init__(f"{first} {second}")

    monkeypatch.setattr(errors, "NeedsTwoArguments", NeedsTwoArguments, raising=False)
    error = _rehydrate("NeedsTwoArguments", "what the child said")
    assert type(error) is SubstrateError and str(error) == "what the child said"


def test_a_worker_runs_its_steps_in_its_own_process_only_when_it_is_told_to(monkeypatch):
    monkeypatch.delenv("SUBSTRATE_WORKER_ISOLATION", raising=False)
    assert default_isolation() == "inline"
    monkeypatch.setenv("SUBSTRATE_WORKER_ISOLATION", "process")
    assert default_isolation() == "process"
    monkeypatch.setenv("SUBSTRATE_WORKER_ISOLATION", "inline")
    assert default_isolation() == "inline"


# -- an answer that is readable text but not JSON ---------------------------------------------------------------------------------------------
from substrate.control.sandbox import ProcessExecutor, StepCrashed  # noqa: E402

from test_control_sandbox import request_for  # noqa: E402


def say_text_that_is_not_json(connection, payload):
    connection.send_bytes(b"this is perfectly good UTF-8 but it is not JSON {")
    connection.close()


def test_an_answer_that_is_valid_text_but_not_json_is_a_crash_not_a_parse_error():
    pipeline, request, _ = request_for()
    with pytest.raises(StepCrashed, match="not a result"):
        ProcessExecutor(target=say_text_that_is_not_json, memory_bytes=None).apply(pipeline, request)
