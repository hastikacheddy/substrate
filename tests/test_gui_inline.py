"""The GUI's endpoints without a server and without threads: what the in-browser demo runs.

`substrate.gui.api` holds what each URL does and what a failure answers; the HTTP server and the browser's web worker are two ways of putting requests to it.
The first group of tests holds the two to the same answers (so the refactor that made the split changed nothing the server says); the rest are about running jobs on the calling
thread, because a browser worker has no other.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from substrate.gui import App, make_server
from substrate.gui import api
from substrate.gui.app import GuiError, JobManager

from conftest import EXPERIMENTS

ROOT = EXPERIMENTS.parent


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return App(ROOT, cache_dir=tmp_path_factory.mktemp("inline-cache"), inline_jobs=True)


@pytest.fixture(scope="module")
def server(app):
    srv = make_server(app, 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def http(server, method, target, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}{target}", method=method,
                                     data=None if body is None else body.encode(), headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# -- the same answers as the server -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["/api/status", "/api/experiments", "/api/experiment?path=proton_transfer.yaml", "/api/references", "/api/job?id=nope",
                                    "/api/job?id=x&since=oops", "/api/experiment?path=..%2Fpyproject.toml", "/api/experiment?path=missing.yaml", "/api/experiment?path=",
                                    "/api/nothing", "/api/transfer?refs=only_one"])
def test_a_get_is_answered_as_the_server_answers_it(app, server, target):
    status, body = http(server, "GET", target)
    answered, text = api.answer(app, "GET", target)
    assert (answered, json.loads(text)) == (status, body)


@pytest.mark.parametrize("target, body", [("/api/run", '{"path": "../pyproject.toml"}'), ("/api/run", '{"path": "nothing.yaml"}'), ("/api/run", "[1, 2]"), ("/api/run", "not json"),
                                          ("/api/transfer/run", '{"refs": ["only_one"]}'), ("/api/transfer/rates", '{"refs": []}'), ("/api/nothing", "{}")])
def test_a_refused_post_is_answered_as_the_server_answers_it(app, server, target, body):
    status, answered = http(server, "POST", target, body)
    code, text = api.answer(app, "POST", target, body)
    assert status >= 400 and (code, json.loads(text)) == (status, answered)


def test_a_finished_run_carries_the_same_result_through_either_door(app, server):
    status, job = http(server, "POST", "/api/run", '{"path": "proton_transfer.yaml", "samples": 20, "seed": 3}')
    assert status == 200
    code, text = api.answer(app, "POST", "/api/run", '{"path": "proton_transfer.yaml", "samples": 20, "seed": 3}')
    assert code == 200
    while job["status"] not in ("done", "error"):                          # the server's job runs on a thread; this one was run before the answer came back
        status, job = http(server, "GET", f"/api/job?id={job['id']}")
    mine = json.loads(text)
    assert mine["status"] == "done" == job["status"]
    assert mine["result"]["input_fingerprint"] == job["result"]["input_fingerprint"] and mine["result"]["ensemble"] == job["result"]["ensemble"]


# -- what only the browser's door does -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["PUT", "DELETE", "HEAD"])
def test_a_method_the_api_does_not_have_is_refused(app, method):
    assert api.answer(app, method, "/api/status")[0] == 405


def test_a_path_outside_the_api_is_not_found(app):
    status, text = api.answer(app, "GET", "/static/app.js")
    assert status == 404 and json.loads(text) == {"error": "not found"}


def test_a_post_without_a_body_is_an_empty_object_not_a_crash(app):
    status, text = api.answer(app, "POST", "/api/run")
    assert status == 400 and "path" in json.loads(text)["error"]


def test_a_bug_inside_a_handler_is_reported_not_swallowed(app, monkeypatch):
    def broken():
        raise RuntimeError("boom")
    monkeypatch.setattr(app, "status", broken)
    status, text = api.answer(app, "GET", "/api/status")
    assert status == 500 and json.loads(text) == {"error": "RuntimeError: boom"}


# -- jobs on the calling thread ------------------------------------------------------------------------------------------------------------
def test_an_inline_job_is_finished_when_submit_returns_and_ran_on_the_calling_thread():
    ran_on = []
    job = JobManager(inline=True).submit("x", lambda log: (log("hello"), ran_on.append(threading.get_ident()), {"n": 1})[2])
    assert ran_on == [threading.get_ident()]
    done = job.to_dict()
    assert done["status"] == "done" and done["result"] == {"n": 1} and [m["text"] for m in done["messages"]] == ["hello"]


def test_an_inline_job_that_fails_records_the_failure_and_the_next_job_still_runs():
    manager = JobManager(inline=True)

    def broken(log):
        raise ValueError("no good")

    bad = manager.submit("bad", broken).to_dict()
    assert bad["status"] == "error" and bad["error"]["type"] == "ValueError" and "no good" in bad["error"]["message"]
    refused = manager.submit("refused", lambda log: (_ for _ in ()).throw(GuiError("not that", 400))).to_dict()
    assert refused["error"] == {"type": "GuiError", "message": "not that"}
    assert manager.submit("fine", lambda log: {"ok": True}).to_dict()["status"] == "done"


def test_the_job_cap_still_holds_inline_and_a_finished_job_frees_the_place():
    manager = JobManager(max_running=1, inline=True)
    manager.submit("one", lambda log: {})
    manager.submit("two", lambda log: {})                                   # the first is finished, so there is room


def test_a_run_through_the_browser_door_starts_no_thread(app):
    before = {t.name for t in threading.enumerate()}
    status, text = api.answer(app, "POST", "/api/run", '{"path": "proton_transfer.yaml", "samples": 10}')
    assert status == 200 and json.loads(text)["status"] == "done"
    assert not [t for t in threading.enumerate() if t.name.startswith("job-") and t.name not in before]


def test_the_default_app_still_runs_its_jobs_on_threads(tmp_path):
    app = App(ROOT, cache_dir=tmp_path)
    assert app.jobs.inline is False
    job = app.jobs.submit("x", lambda log: {"n": 1}, wait=True)
    assert job.to_dict()["status"] == "done"
