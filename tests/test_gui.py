"""The GUI's server and application layer, over real HTTP on a free local port (no browser needed).

The page itself is exercised by eye (and by a syntax check here); what is tested is everything it depends on: what the API returns, that
what it returns is what the engines produced, that edits move the numbers the right way, that refusals keep the stages that completed,
and that the server refuses requests it should not serve.
"""
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from substrate.gui import App, GuiError, make_server, serve
from substrate.gui import serialize as ser
from substrate.gui.app import JobManager
from substrate.gui.server import STATIC

from conftest import EXPERIMENTS

ROOT = EXPERIMENTS.parent


# -- a running server ----------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return App(ROOT, cache_dir=tmp_path_factory.mktemp("gui-cache"))


@pytest.fixture(scope="module")
def server(app):
    srv = make_server(app, 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(server, path, body=None, headers=None, raw=None, method=None):
    port = server.server_address[1]
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})} if data is not None else (headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def get(server, path, **kw):
    status, body, headers = call(server, path, **kw)
    return status, (json.loads(body) if "json" in headers.get("Content-Type", "") else body)


def finish(server, job, timeout=120):
    """Poll a job to completion, as the page does."""
    deadline = time.time() + timeout
    while job["status"] not in ("done", "error"):
        assert time.time() < deadline, "job did not finish"
        time.sleep(0.05)
        _, job = get(server, f"/api/job?id={job['id']}")
    return job


def run(server, path, **body):
    status, job = get(server, "/api/run", body={"path": path, **body})
    assert status == 200, job
    done = finish(server, job)
    assert done["status"] == "done", done["error"]
    return done["result"]


# -- the API -------------------------------------------------------------------------------------------------------------------------------
def test_status_and_the_experiment_listing(server):
    status, st = get(server, "/api/status")
    assert status == 200 and st["version"] == "0.1.0" and st["experiments"] >= 14 and isinstance(st["pyscf"], bool)
    _, listing = get(server, "/api/experiments")
    by_path = {e["path"]: e for e in listing["experiments"]}
    assert by_path["proton_transfer_molecular.yaml"]["needs_qc"] is False and by_path["proton_transfer_molecular.yaml"]["scale"] == "electronic_structure"
    assert by_path["references/zundel_hf.yaml"]["needs_qc"] is True and by_path["references/zundel_hf.yaml"]["group"] == "references"
    assert not any("invalid" in e for e in listing["experiments"])


def test_an_experiment_reports_its_inputs_and_its_description(server):
    _, info = get(server, "/api/experiment?path=proton_transfer_molecular.yaml")
    assert info["kind"] == "electronic.evb_two_state_2d" and info["propagation"][-1] == "reaction"
    assert info["parameters"]["coupling"] == {"value": 0.6, "unit": "eV", "sigma": 0.03, "source": "input"}
    assert "flexible" in info["name"] and "Proton transfer through the molecular scale" in info["header"]
    _, qc = get(server, "/api/experiment?path=references/fhf_hf.yaml")
    assert qc["needs_qc"] and qc["structure"]["molecule"] == {"template": "bifluoride_anion"}


@pytest.mark.parametrize("path", ["../pyproject.toml", "..%2Fpyproject.toml", "/etc/passwd", "references/../../README.md", "C:\\Windows\\win.ini",
                                  "proton_transfer.txt", "nope.yaml"])
def test_only_experiment_files_inside_the_experiments_folder_can_be_read(server, path):
    status, body = get(server, f"/api/experiment?path={path}")
    assert status in (400, 403, 404) and "error" in body


# -- running -------------------------------------------------------------------------------------------------------------------------------------
def test_a_run_returns_every_stage_with_its_plots_and_an_unbroken_lineage(server):
    result = run(server, "proton_transfer_electronic.yaml", samples=0)
    assert result["ok"] and [s["scale"] for s in result["stages"]] == ["electronic_structure", "quantum", "reaction"]
    electronic, quantum, reaction = (s["system"] for s in result["stages"])
    line = electronic["plots"][0]
    assert line["type"] == "line" and len(line["x"]) == 241 and {s["name"] for s in line["series"]} >= {"ground state", "reactant diabat"}
    assert quantum["plots"][0]["hlines"] and quantum["plots"][0]["series"][0]["name"] == "potential"
    assert [s["name"] for s in reaction["plots"][0]["series"]] == ["A", "B"] and reaction["observables"]["relaxation_time"]["unit"] == "s"
    previous = None                                                                              # each hop starts from what the last one produced
    for stage in result["stages"]:
        if previous is not None:
            assert stage["translate"]["input"] == previous and stage["solve"]["input"] == stage["translate"]["output"]
        previous = stage["solve"]["output"]
    assert result["stages"][1]["translate"]["approximations"] and result["input_fingerprint"] == result["stages"][0]["solve"]["input"]


def test_the_plotted_curve_is_what_the_engine_computed(server, app):
    from substrate import load_experiment
    exp = load_experiment(EXPERIMENTS / "proton_transfer_electronic.yaml")
    truth = app.pipeline.run(exp.system, exp.propagation).trace[0].obs("scan_energy")
    shown = run(server, "proton_transfer_electronic.yaml", samples=0)["stages"][0]["system"]["plots"][0]["series"][0]["y"]
    assert np.allclose(shown, truth, atol=1e-5)                                                  # rounded to 5 decimals for transport


def test_editing_an_input_moves_the_outputs_the_right_way(server):
    base = run(server, "proton_transfer_molecular.yaml", samples=0)
    stronger = run(server, "proton_transfer_molecular.yaml", samples=0, overrides={"coupling": {"value": 0.75, "sigma": None}})
    barrier = lambda r: r["stages"][0]["system"]["observables"]["classical_barrier"]["value"]
    assert barrier(stronger) < barrier(base)                                                      # more coupling, lower barrier
    assert stronger["ok"] and stronger["overrides"] == {"coupling": {"value": 0.75, "sigma": None}}
    assert stronger["input_fingerprint"] != base["input_fingerprint"]
    assert run(server, "proton_transfer_molecular.yaml", samples=0)["input_fingerprint"] == base["input_fingerprint"]    # an edit does not leak


def test_an_ensemble_attaches_spread_to_the_outputs(server):
    result = run(server, "proton_transfer_molecular.yaml", samples=16, seed=3)
    assert result["ensemble"]["n_requested"] == 16 and result["ensemble"]["n_ok"] > 0 and result["seed"] == 3
    spread = [q for st in result["stages"] for q in st["system"]["observables"].values() if q.get("sigma")]
    assert spread, "no observable carries an uncertainty after an ensemble"
    assert run(server, "proton_transfer_molecular.yaml", samples=0)["ensemble"] is None


def test_a_refusal_by_the_physics_keeps_the_stages_that_completed(server):
    result = run(server, "proton_transfer_molecular.yaml", samples=0, overrides={"oo_equilibrium": {"value": 2.3}})
    assert result["ok"] is False and result["error"]["type"] == "ValidationError" and result["overrides"] == {"oo_equilibrium": {"value": 2.3}}
    assert result["failed"]["scale"] == "molecular" and "fewer than two wells" in result["failed"]["message"]
    assert [s["scale"] for s in result["stages"]] == ["electronic_structure"] and result["stages"][0]["system"]["plots"]     # what ran is still shown
    assert [p["scale"] for p in result["plan"] if p["kind"] == "solve"] == ["electronic_structure", "molecular", "reaction"]


@pytest.mark.parametrize("overrides, message", [
    ({"nope": 1.0}, "no parameter 'nope'"), ({"coupling": {"value": "x"}}, "finite number"), ({"coupling": {"sigma": -1}}, "must not be negative"),
    ({"coupling": {"value": 1e999}}, "finite number"),
])
def test_bad_edits_are_refused_before_anything_runs(app, overrides, message):
    with pytest.raises(GuiError, match=message):
        app.run("proton_transfer_molecular.yaml", overrides=overrides, samples=0)


def test_the_ensemble_size_is_bounded(app):
    with pytest.raises(GuiError, match="between 0 and 2000"):
        app.run("proton_transfer_molecular.yaml", samples=5000)


# -- jobs ----------------------------------------------------------------------------------------------------------------------------------------------
def test_a_job_reports_messages_errors_and_unknown_ids():
    manager = JobManager()
    ok = manager.submit("fine", lambda log: (log("one"), log("two"), {"x": np.float64(1.5), "bad": float("nan")})[2], wait=True)
    assert ok.to_dict()["status"] == "done" and [m["text"] for m in ok.to_dict()["messages"]] == ["one", "two"]
    assert ok.to_dict()["result"] == {"x": 1.5, "bad": None}                                      # numpy scalars and NaN are made JSON-safe
    assert [m["text"] for m in ok.to_dict(since=1)["messages"]] == ["two"]
    broken = manager.submit("broken", lambda log: 1 / 0, wait=True).to_dict()
    assert broken["status"] == "error" and broken["error"]["type"] == "ZeroDivisionError" and broken["result"] is None
    refused = manager.submit("refused", lambda log: (_ for _ in ()).throw(GuiError("no")), wait=True).to_dict()
    assert refused["error"] == {"type": "GuiError", "message": "no"}
    with pytest.raises(GuiError) as error:
        manager.get("nope")
    assert error.value.status == 404


# -- the server refuses what it should ---------------------------------------------------------------------------------------------------------
def test_requests_for_another_host_or_from_another_origin_are_refused(server):
    assert call(server, "/api/status", headers={"Host": "evil.example:80"})[0] == 403             # DNS rebinding
    assert call(server, "/api/status", headers={"Origin": "http://evil.example"})[0] == 403        # another page in the browser
    assert call(server, "/api/run", body={"path": "proton_transfer.yaml"}, headers={"Origin": "http://evil.example"})[0] == 403
    port = server.server_address[1]
    assert call(server, "/api/status", headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200  # its own page is fine


def test_posts_must_be_small_valid_json_objects(server):
    assert call(server, "/api/run", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 415   # a form post, not our page
    assert call(server, "/api/run", raw=b"{not json")[0] == 400
    assert call(server, "/api/run", raw=b"[1, 2]")[0] == 400
    try:                                                                                          # refused without being read: the client may see 413 or a reset
        assert call(server, "/api/run", raw=b"{" + b" " * 1_100_000 + b"}")[0] == 413
    except (ConnectionError, urllib.error.URLError):
        pass
    assert call(server, "/api/nope", body={})[0] == 404
    assert call(server, "/api/run", body={"path": "../x.yaml"})[0] in (400, 403, 404)


def test_only_the_pages_own_files_are_served_and_with_a_strict_policy(server):
    status, body, headers = call(server, "/")
    assert status == 200 and b"mission control" in body and "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["Cache-Control"] == "no-store"
    assert b"<script>" not in body                                                                   # no inline script: the policy would block it
    for name, kind in (("app.js", "javascript"), ("charts.js", "javascript"), ("app.css", "text/css")):
        status, body, headers = call(server, f"/static/{name}")
        assert status == 200 and kind in headers["Content-Type"] and body
    for bad in ("/static/../app.py", "/static/..%2Fapp.py", "/static/sub/x.js", "/static/nothing.js", "/server.py"):
        assert call(server, bad)[0] == 404


def test_a_port_that_is_taken_is_reported_not_crashed(capsys):
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        assert serve(blocker.getsockname()[1], open_browser=False) == 1
    assert "cannot listen" in capsys.readouterr().out


def test_the_pages_scripts_are_syntactically_valid():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    for name in ("app.js", "charts.js"):
        result = subprocess.run([node, "--check", str(STATIC / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


# -- serialisation --------------------------------------------------------------------------------------------------------------------------------
def test_clean_makes_everything_json_safe():
    out = ser.clean({"a": np.array([1.0, np.nan, np.inf]), "b": (np.int64(3), np.bool_(True)), "c": {"d": np.float32(0.5)}})
    assert out == {"a": [1.0, None, None], "b": [3, True], "c": {"d": 0.5}}
    json.dumps(out, allow_nan=False)


def test_quantities_keep_uncertainty_and_summarise_arrays():
    from substrate import Quantity
    q = ser.quantity(Quantity(2.0, "eV", 0.1, "x", (1.9, 2.1)))
    assert q == {"unit": "eV", "value": 2.0, "sigma": 0.1, "band": [1.9, 2.1], "source": "x"}
    a = ser.quantity(Quantity(np.linspace(-1, 3, 50), "angstrom"))
    assert a["array"] == {"shape": [50], "min": -1.0, "max": 3.0}
    assert ser.quantity(Quantity(np.array([1.0, 2.0]), "1"))["array"]["values"] == [1.0, 2.0]


# -- the transfer study (synthetic references, so no PySCF is needed) ----------------------------------------------------------------------------
SYNTHETIC = """experiment:
  id: SYN-{name}
  phenomenon: synthetic reference
  system:
    name: synthetic {name}
    scale: electronic_structure
    kind: electronic.evb_two_state_2d
    parameters:
      morse_depth: {{value: 4.6, unit: eV}}
      morse_alpha: {{value: 2.2, unit: 1/angstrom}}
      morse_r_eq: {{value: 0.96, unit: angstrom}}
      coupling: {{value: {coupling}, unit: eV}}
      coupling_decay: {{value: 3.0, unit: 1/angstrom}}
      reference_distance: {{value: 2.5, unit: angstrom}}
      oo_depth: {{value: 0.4, unit: eV}}
      oo_alpha: {{value: 2.5, unit: 1/angstrom}}
      oo_equilibrium: {{value: 2.7, unit: angstrom}}
      diabatic_offset: {{value: 0.0, unit: eV}}
      particle_mass: {{value: 1.007276, unit: amu}}
      heavy_atom_mass: {{value: 15.9949, unit: amu}}
      temperature: {{value: 300, unit: K}}
      x_extent: {{value: 0.65, unit: angstrom}}
      distance_min: {{value: 2.3, unit: angstrom}}
      distance_max: {{value: 3.0, unit: angstrom}}
      n_x: {{value: 21, unit: "1"}}
      n_r: {{value: 21, unit: "1"}}
  propagation: [electronic_structure]
{hint}"""


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    refs = root / "experiments" / "references"
    refs.mkdir(parents=True)
    (refs / "syn_a.yaml").write_text(SYNTHETIC.format(name="A", coupling=0.6, hint=""))
    (refs / "syn_b.yaml").write_text(SYNTHETIC.format(name="B", coupling=0.45, hint="  calibration: {window_ev: 2.0}" + chr(10)))
    (root / "experiments" / "plain.yaml").write_text(SYNTHETIC.format(name="P", coupling=0.6, hint=""))
    return App(root, cache_dir=root / "cache")


def test_the_transfer_payload_has_every_pair_and_a_perfect_diagonal(study):
    logs = []
    payload = study.run_transfer(["syn_a", "syn_b"], log=logs.append)
    assert payload["names"] == ["syn_a", "syn_b"] and len(payload["refs"]) == 2 and set(payload["pairs"]) == {"syn_a|syn_a", "syn_a|syn_b", "syn_b|syn_a", "syn_b|syn_b"}
    m = payload["matrix"]
    assert m["rmse"][0][0] < 1e-4 and m["rmse"][1][1] < 1e-4 and m["rmse"][0][1] > 0.01 and m["rmse"][1][0] > 0.01           # exact recovery; a weaker coupling shows
    ref = payload["refs"][0]
    assert len(ref["e"]) == len(ref["x"]) == 21 and len(ref["e"][0]) == len(ref["r"]) == 21 and ref["calibration"]["pinned"] == {}
    pair = payload["pairs"]["syn_a|syn_b"]
    assert len(pair["model"]) == 21 and len(pair["model"][0]) == 21 and pair["barrier_error"] == pytest.approx(m["barrier"][0][1])
    assert payload["refs"][0]["baselines"]["constant_rmse"] > 0.05 and any("fitting the model" in l for l in logs)
    json.dumps(payload, allow_nan=False)


def test_a_transfer_result_is_saved_and_reloaded_unless_recompute_is_asked_for(study):
    first = study.run_transfer(["syn_a", "syn_b"])
    assert study.cached_transfer(["syn_b", "syn_a"]) == study.cached_transfer(["syn_a", "syn_b"]) is not None          # order does not matter
    logs = []
    again = study.run_transfer(["syn_a", "syn_b"], log=logs.append)
    assert again == first and any("loaded the saved result" in l for l in logs)
    logs.clear()
    study.run_transfer(["syn_a", "syn_b"], recompute=True, log=logs.append)
    assert not any("loaded the saved result" in l for l in logs)
    path = study.experiments_dir / "references" / "syn_a.yaml"
    path.write_text(path.read_text() + "\n# edited\n")                                                              # a changed file is a different study
    assert study.cached_transfer(["syn_a", "syn_b"]) is None
    path.write_text(path.read_text().replace("\n# edited\n", ""))


def test_transfer_requests_are_validated(study):
    for names in (None, ["syn_a"], ["syn_a", "syn_a"], ["syn_a", "../x"], ["syn_a", "missing"], ["a"] * 30):
        with pytest.raises(GuiError):
            study.run_transfer(names)
    with pytest.raises(GuiError, match="between 0.02 and 1.5"):
        study.run_rates(["syn_a", "syn_b"], 9.0)
    with pytest.raises(GuiError, match="prior widths"):
        study.run_learning(["syn_a", "syn_b"], "syn_b", "syn_a", [0.0], [1])
    with pytest.raises(GuiError, match="among the chosen"):
        study.run_learning(["syn_a", "syn_b"], "syn_c", None, [0.3], [1])


def test_the_rate_comparison_is_right_on_the_diagonal_and_off_it(study):
    result = study.run_rates(["syn_a", "syn_b"], 0.4)
    own = result["cells"]["syn_a|syn_a"]["ratio"], result["cells"]["syn_b|syn_b"]["ratio"]
    assert all(0.9 < r < 1.1 for r in own)                                                          # a calibration reproduces its own reference's rate
    assert result["cells"]["syn_a|syn_b"]["ratio"] != pytest.approx(1.0, abs=0.05)                  # but a different coupling is visible
    assert result["real"]["syn_a"]["k"] > 0 and result["real"]["syn_a"]["distance"] > 2.3 and result["real"]["syn_a"]["kie"] > 1


def test_the_learning_curve_payload(study):
    result = study.run_learning(["syn_a", "syn_b"], "syn_a", "syn_b", [0.3], [0, 1, 3])               # syn_a: a 1.5 eV window, so one distance is too few
    assert result["counts"] == [0, 1, 3] and set(result["priors"]) == {"0.3"} and result["floor"] < 0.01
    prior = {p["k"]: p for p in result["priors"]["0.3"]}
    scratch = {p["k"]: p for p in result["scratch"]}
    assert set(prior) == {0, 1, 3} and set(scratch) == {1, 3}
    assert prior[0]["rmse"] > 0.01 and prior[3]["rmse"] < prior[0]["rmse"]                          # data beat the unchanged transferred parameters
    assert scratch[1]["note"] and scratch[1]["rmse"] is None                                        # one distance cannot be fitted from scratch, and it says so


def test_a_quantum_chemistry_reference_is_refused_clearly_when_pyscf_is_missing(study, monkeypatch):
    qc = study.experiments_dir / "references" / "qc_x.yaml"
    qc.write_text(SYNTHETIC.format(name="Q", coupling=0.6, hint="").replace("electronic.evb_two_state_2d", "electronic.qc_scan_2d"))
    monkeypatch.setattr(study, "_pyscf", False)
    with pytest.raises(GuiError, match="PySCF is not available") as error:
        study.reference("qc_x")
    assert error.value.status == 503


def test_each_reference_in_the_payload_carries_its_own_fit_window(study):
    payload = study.run_transfer(["syn_a", "syn_b"])
    assert [r["window"] for r in payload["refs"]] == [1.5, 2.0] and [r["calibration"]["rmse"].get("2") is not None for r in payload["refs"]] == [False, True]
    pair = payload["pairs"]["syn_a|syn_b"]
    assert pair["rmse"]["2"] == pytest.approx(payload["matrix"]["rmse"][0][1])                       # the matrix cell is the target's 2 eV number
    assert payload["matrix"]["rmse"][0][0] == pytest.approx(payload["pairs"]["syn_a|syn_a"]["rmse"]["1.5"])
    assert payload["refs"][1]["baselines"]["constant_rmse"] > payload["refs"][0]["baselines"]["constant_rmse"] * 0.9


def test_a_study_of_every_shipped_reference_is_allowed(app):
    from substrate.gui.app import MAX_REFERENCES
    names = [r["name"] for r in app.references_listing()]
    assert len(names) >= 15 and not any(r.get("invalid") for r in app.references_listing())
    assert app._check_names(names) == names and len(names) <= MAX_REFERENCES          # the page selects them all by default
