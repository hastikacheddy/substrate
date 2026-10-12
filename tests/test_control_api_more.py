"""Cases of the API that the first tests did not pin, found by mutation testing: the body-size guard at its exact boundaries (driven directly as an ASGI application, where a request can
declare a size it never sends), and the schema limits on request bodies, whose refusals are told apart from the later checks by what they say."""
import asyncio

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from substrate.control.api import BodyLimit  # noqa: E402

from test_control_api import create_experiment, h, world  # noqa: E402,F401


# -- BodyLimit, as an ASGI application ----------------------------------------------------------------------------------------------------
def run_through(app, *, declared=None, chunks=(), limit=100):
    """Send one request through BodyLimit(app) and return (what the inner application saw, what was sent back)."""
    headers = [(b"content-length", str(declared).encode())] if declared is not None else []
    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    queue = list(chunks)
    sent = []

    async def receive():
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(BodyLimit(app, limit=limit)(scope, receive, send))
    return sent


class Inner:
    """An application that reads the whole body, says how much it was, and answers 200."""
    def __init__(self, answer_first=False):
        self.read, self.called, self.answer_first = 0, False, answer_first

    async def __call__(self, scope, receive, send):
        self.called = True
        if self.answer_first:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"early", "more_body": True})
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            self.read += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        if not self.answer_first:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})


def statuses(sent):
    return [m["status"] for m in sent if m["type"] == "http.response.start"]


@pytest.mark.parametrize("declared, accepted", [(0, True), (99, True), (100, True), (101, False), (150, False), (10_000, False)])
def test_a_declared_length_is_accepted_up_to_and_including_the_limit(declared, accepted):
    inner = Inner()
    sent = run_through(inner, declared=declared, chunks=[b"x" * min(declared, 100)] if declared else [])
    assert (statuses(sent) == [200]) is accepted and inner.called is accepted
    if not accepted:
        assert statuses(sent) == [413]


def test_a_length_that_is_not_a_number_is_refused_without_running_the_application():
    inner = Inner()
    for bad in ("abc", "-1", "1e3", " 5", ""):
        sent = run_through(inner, declared=bad)
        assert statuses(sent) == [413] and not inner.called, bad


@pytest.mark.parametrize("sizes, accepted", [([100], True), ([60, 40], True), ([50, 50, 1], False), ([101], False), ([30, 30, 30, 30], False), ([1] * 100, True), ([1] * 101, False)])
def test_a_body_of_unknown_length_is_accepted_up_to_exactly_the_limit_as_it_streams(sizes, accepted):
    inner = Inner()
    sent = run_through(inner, chunks=[b"x" * n for n in sizes])
    assert (statuses(sent) == [200]) is accepted
    if not accepted:
        assert statuses(sent) == [413]


def test_a_refusal_after_the_application_has_begun_to_answer_is_not_a_second_answer():
    inner = Inner(answer_first=True)
    sent = run_through(inner, chunks=[b"x" * 60, b"x" * 60])
    assert statuses(sent) == [200]                                                          # the application's own start stands; nothing further is added to it
    assert [m["type"] for m in sent].count("http.response.start") == 1


# -- the schema limits --------------------------------------------------------------------------------------------------------------------
def test_a_transfer_study_needs_at_least_two_references_by_the_schema_itself(world):
    c = world["client"]
    for references in ([], ["only-one"]):
        response = c.post("/transfers", json={"references": references}, headers=h(world, "ada"))
        assert response.status_code == 422 and "at least 2" in response.text, response.text


@pytest.mark.parametrize("field, value, needle", [("window_ev", 0, "greater than 0"), ("window_ev", -1.0, "greater than 0"), ("window_ev", 51, "less than or equal to 50"),
                                                   ("sigma_ev", 0, "greater than 0"), ("sigma_ev", 11, "less than or equal to 10")])
def test_the_energy_window_and_the_prior_width_of_a_transfer_study_are_bounded_by_the_schema(world, field, value, needle):
    response = world["client"].post("/transfers", json={"references": ["EXP-000001", "EXP-000002"], field: value}, headers=h(world, "ada"))
    assert response.status_code == 422 and needle in response.text, response.text


def test_a_query_for_the_data_fabric_has_at_most_ten_keys_by_the_schema(world):
    response = world["client"].post("/datasets/ingest", json={"adapter": "nist.ion_energetics", "query": {f"k{i}": 1 for i in range(11)}}, headers=h(world, "ada"))
    assert response.status_code == 422 and "at most 10" in response.text, response.text
    ok = world["client"].post("/datasets/ingest", json={"adapter": "nist.ion_energetics", "query": {}}, headers=h(world, "ada"))
    assert ok.status_code == 201


# -- authentication scheme and the route table ---------------------------------------------------------------------------------------------
def test_only_the_bearer_scheme_carries_a_key(world):
    c, token = world["client"], world["tokens"]["ada"]
    assert c.get("/experiments", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert c.get("/experiments", headers={"Authorization": f"bearer {token}"}).status_code == 200            # the scheme name is not case sensitive
    for header in (f"Basic {token}", f"Token {token}", f"Digest {token}", token, f"Bearer", f"Bearer  ", f"Bearer {token} extra"):
        assert c.get("/experiments", headers={"Authorization": header}).status_code == 401, header


WRITES = [("post", "/experiments", {"name": "x", "spec": {}}), ("post", "/runs", {"experiment_id": "EXP-000001"}), ("post", "/runs/RUN-000001/cancel", None),
          ("post", "/reproduce/RUN-000001", None), ("post", "/transfers", {"references": ["EXP-000001", "EXP-000002"]}),
          ("post", "/datasets/ingest", {"adapter": "nist.ion_energetics", "query": {}})]


@pytest.mark.parametrize("method, path, body", WRITES)
def test_a_viewer_cannot_use_any_route_that_writes(world, method, path, body):
    response = getattr(world["client"], method)(path, headers=h(world, "vera"), **({"json": body} if body is not None else {}))
    assert response.status_code == 403, (path, response.text)


@pytest.mark.parametrize("method, path, body", WRITES)
def test_an_operator_cannot_use_a_route_that_writes_except_to_cancel(world, method, path, body):
    response = getattr(world["client"], method)(path, headers=h(world, "olga"), **({"json": body} if body is not None else {}))
    assert (response.status_code != 403) is path.endswith("/cancel"), (path, response.text)               # cancel is theirs: it gets as far as looking the run up


def test_an_operator_watches_runs_but_cannot_list_or_download_what_they_produced(world):
    exp = create_experiment(world)["id"]
    run_id = world["client"].post("/runs", json={"experiment_id": exp, "n_samples": 0}, headers=h(world, "ada")).json()["id"]
    from substrate.control.worker import Worker
    Worker(world["orch"]).run_until_idle()
    sha = world["client"].get(f"/runs/{run_id}/artifacts", headers=h(world, "ada")).json()["artifacts"][0]["sha256"]
    assert world["client"].get(f"/runs/{run_id}", headers=h(world, "olga")).status_code == 200
    assert world["client"].get(f"/runs/{run_id}/artifacts", headers=h(world, "olga")).status_code == 403
    assert world["client"].get(f"/artifacts/{sha}", headers=h(world, "olga")).status_code == 403
    assert world["client"].get(f"/runs/{run_id}/artifacts", headers=h(world, "vera")).status_code == 200
    assert world["client"].get(f"/artifacts/{sha}", headers=h(world, "vera")).status_code == 200


def test_the_audit_trail_returns_the_newest_events_when_it_is_cut_short(world):
    for n in range(6):
        world["registry"].audit(world["tenant"], "test.event", n=n)
    everything = world["client"].get("/audit?limit=1000", headers=h(world, "olga")).json()["events"]
    last_two = world["client"].get("/audit?limit=2", headers=h(world, "olga")).json()["events"]
    assert len(everything) >= 6 and last_two == everything[-2:] and last_two[-1]["payload"] == {"n": 5} and last_two[0]["seq"] < last_two[1]["seq"]
