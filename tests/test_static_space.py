"""The static Hugging Face Space: the free way to publish Mission control, with the Python running in the visitor's browser.

The builder ships what git tracks and a scanned copy of the energy cache; the uploader accepts a static folder as well as a Docker one. Nothing here talks to the network or to a browser
(the page was driven by hand in one): what is checked is the folder, the bundle in it, and that the study in the bundle is one the GUI will actually find for the bundle's own files.
"""
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "deploy" / "huggingface"
REAL_CACHE = Path.home() / ".cache" / "substrate"


def load(name):
    sys.path.insert(0, str(HF))                                         # the static builder imports its sibling, build_space
    try:
        spec = importlib.util.spec_from_file_location(name, HF / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HF))
    return module


static = load("build_static_space")
upload = load("upload_space")
needs_git = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="the builder ships what git tracks, so it needs a checkout")
SPACE = "Hastika06/substrate-mission-control"


@pytest.fixture
def cache(tmp_path):
    folder = tmp_path / "cache"
    folder.mkdir()
    with sqlite3.connect(folder / "qc.sqlite3") as db:
        db.execute("CREATE TABLE energies (key TEXT PRIMARY KEY, payload TEXT)")
        db.executemany("INSERT INTO energies VALUES (?, ?)", [("aa11", json.dumps({"energy": -1.5, "converged": True, "program": "pyscf 2.14.0"})),
                                                              ("bb22", json.dumps({"energy": -2.5, "converged": True, "program": "pyscf 2.14.0"}))])
    return folder


@pytest.fixture
def built(tmp_path, cache):
    out = tmp_path / "static-space"
    summary = static.build(out, SPACE, cache, prewarm=False)
    return out, summary


# -- the folder ------------------------------------------------------------------------------------------------------------------------------
@needs_git
def test_the_folder_holds_the_page_the_shim_the_worker_the_bundle_and_the_card(built):
    out, summary = built
    names = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())
    assert names == ["README.md", "SOURCE_COMMIT", "bundle.zip", "index.html", "static/app.css", "static/app.js", "static/charts.js", "static/web_shim.js", "worker.js"]
    assert summary["energies"] == 2 and summary["references"] >= 20 and summary["study_precomputed"] is False and summary["pyodide"] == static.PYODIDE_VERSION


@needs_git
def test_the_page_is_the_guis_own_with_the_shim_loaded_before_the_scripts_that_call_the_api(built):
    out, _ = built
    own = (ROOT / "src" / "substrate" / "gui" / "static" / "index.html").read_text(encoding="utf-8")
    page = (out / "index.html").read_text(encoding="utf-8")
    assert page.replace(static.SHIM_TAG, "") == own.replace("\r\n", "\n")
    assert page.index("web_shim.js") < page.index("charts.js") < page.index("app.js")
    for name in ("app.js", "app.css", "charts.js"):
        assert (out / "static" / name).read_bytes() == (ROOT / "src" / "substrate" / "gui" / "static" / name).read_bytes()


def test_a_page_that_is_not_the_one_expected_is_refused_rather_than_patched_by_guesswork():
    with pytest.raises(static.common.BuildError, match="charts.js"):
        static.page("<html><body>nothing to hook into</body></html>")
    with pytest.raises(static.common.BuildError, match="charts.js"):
        static.page('<script src="/static/charts.js"></script><script src="/static/charts.js"></script>')


@needs_git
def test_the_bundle_holds_the_source_the_experiments_and_the_energies_and_nothing_machine_made(built, tmp_path):
    out, _ = built
    with zipfile.ZipFile(out / "bundle.zip") as bundle:
        names = bundle.namelist()
        assert {"src/substrate/gui/api.py", "src/substrate/gui/app.py", "experiments/proton_transfer.yaml", "cache/qc.sqlite3"} <= set(names)
        assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc") or n.startswith("cache/gui") or n.endswith("prewarm.py")]
        assert not [n for n in names if n.split("/")[0] not in ("src", "experiments", "cache")]
        bundle.extract("cache/qc.sqlite3", tmp_path / "unpacked")
    with sqlite3.connect(tmp_path / "unpacked" / "cache" / "qc.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM energies").fetchone()[0] == 2


@needs_git
def test_the_same_commit_and_cache_give_the_same_bundle_bytes(tmp_path, cache):
    one, two = tmp_path / "one", tmp_path / "two"
    static.build(one, SPACE, cache, prewarm=False)
    static.build(two, SPACE, cache, prewarm=False)
    assert (one / "bundle.zip").read_bytes() == (two / "bundle.zip").read_bytes()


@needs_git
def test_the_card_is_a_static_space_with_the_license_left_open_and_says_what_it_cannot_do(built):
    out, _ = built
    card = (out / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(card.split("---")[1])
    assert front["sdk"] == "static" and front["license"] == "other" and "app_port" not in front and len(front["short_description"]) <= 60
    for said in ("in your browser", "Pyodide", "No new quantum chemistry", "PySCF cannot run here", "2 energies", "no precomputed study", "Recomputing can give slightly different numbers",
                 static.PYODIDE_NUMERICS, "docs/limits.md"):
        assert said in card or said.replace("2 energies", "2 energies") in card, said


@needs_git
def test_nothing_that_shows_this_machine_is_in_the_text_files_of_the_folder(built):
    out, _ = built
    markers = static.common.machine_markers()
    public = (static.common.GITHUB, SPACE, SPACE.split("/")[0])         # the author's public names belong on the card; the account name on this machine is a part of one of them
    for path in out.rglob("*"):
        if path.is_file() and path.suffix != ".zip":
            text = path.read_text(encoding="utf-8")
            for name in public:
                text = text.replace(name, "")
            assert not [m for m in markers if m in text.lower()], path.name


def test_the_card_names_the_native_numerics_only_when_a_study_was_precomputed():
    with_study = static.card(SPACE, "a" * 40, 10292, 24, "numpy 2.2.6 and scipy 1.14.0")
    assert "computed ahead of time (natively, with numpy 2.2.6 and scipy 1.14.0)" in with_study and "10,292 energies" in with_study and "no precomputed study" not in with_study
    assert "There is no precomputed study" in static.card(SPACE, "a" * 40, 10292, 24)


# -- what it refuses -----------------------------------------------------------------------------------------------------------------------------
@needs_git
def test_it_never_writes_into_a_folder_that_has_something_in_it(tmp_path, cache):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "mine.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(static.common.BuildError, match="not empty"):
        static.build(out, SPACE, cache, prewarm=False)
    assert (out / "mine.txt").read_text(encoding="utf-8") == "keep"


@needs_git
@pytest.mark.parametrize("poison", ["localhost", "127.0.0.1", "c:/", "/home/someone"])
def test_a_cache_value_that_shows_this_machine_stops_the_build_and_leaves_nothing_behind(tmp_path, cache, poison):
    with sqlite3.connect(cache / "qc.sqlite3") as db:
        db.execute("INSERT INTO energies VALUES (?, ?)", ("cc33", json.dumps({"energy": -3.0, "note": f"computed at {poison}/work"})))
    out = tmp_path / "out"
    with pytest.raises(static.common.BuildError, match="would publish something from this machine"):
        static.build(out, SPACE, cache, prewarm=False)
    assert list(out.iterdir()) == []


@needs_git
def test_a_missing_cache_is_an_error_not_an_empty_space(tmp_path):
    with pytest.raises(static.common.BuildError, match="no energy cache"):
        static.build(tmp_path / "out", SPACE, tmp_path / "nowhere", prewarm=False)
    assert list((tmp_path / "out").iterdir()) == []


def test_a_bad_space_id_is_refused_before_anything_is_written(tmp_path, cache):
    with pytest.raises(static.common.BuildError, match="not a Space id"):
        static.build(tmp_path / "out", "no-slash", cache, prewarm=False)
    assert not (tmp_path / "out").exists()


def test_the_command_line_reports_a_refusal_and_exits_nonzero(tmp_path, capsys):
    assert static.main([str(tmp_path / "out"), "--space-id", SPACE, "--cache-dir", str(tmp_path / "nowhere")]) == 1
    assert "no energy cache" in capsys.readouterr().err


# -- the page and the worker ---------------------------------------------------------------------------------------------------------------------
def test_the_worker_loads_a_pinned_pyodide_and_the_bundle_the_builder_writes_and_the_shim_starts_that_worker():
    worker = (HF / "static_space" / "worker.js").read_text(encoding="utf-8")
    shim = (HF / "static_space" / "web_shim.js").read_text(encoding="utf-8")
    assert f"/pyodide/v{static.PYODIDE_VERSION}/full/" in worker and "latest" not in worker
    assert "new Worker('worker.js')" in shim and "/api/" in shim
    # a private Space's host authorises the page's requests and refuses a worker's (a 401, seen on Hugging Face): the page fetches the bundle and hands it over
    assert "fetch(" not in worker.replace("loadPyodide", "") and "importScripts(" in worker                    # the worker requests nothing itself but the pinned CDN script
    assert "realFetch('bundle.zip')" in shim and "type: 'bundle'" in shim and "data.type === 'bundle'" in worker
    for package in ("numpy", "scipy", "pyyaml", "sqlite3"):
        assert f"'{package}'" in worker
    assert "SUBSTRATE_MAX_ENSEMBLE" in worker and "inline_jobs=True" in worker


@pytest.mark.parametrize("name", ["worker.js", "web_shim.js"])
def test_the_scripts_are_valid_javascript(name):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run([node, "--check", str(HF / "static_space" / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@needs_git
@pytest.mark.skipif(not (REAL_CACHE / "qc.sqlite3").is_file(), reason="this checks the real energy cache; there is none on this machine")
def test_the_study_in_the_bundle_is_the_one_the_gui_looks_up_for_the_bundles_own_files(tmp_path):
    """The point of computing the study at build time: a saved study is keyed to the bytes of the reference files, so one made from other bytes would be ignored and recomputed."""
    out = tmp_path / "static-space"
    summary = static.build(out, SPACE, REAL_CACHE)                      # the real thing, study included: this takes about a minute
    assert summary["study_precomputed"] is True
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(out / "bundle.zip") as bundle:
        bundle.extractall(unpacked)
    assert len(list((unpacked / "cache" / "gui").glob("transfer-*.json"))) == 1
    code = ("import sys, os; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1] + '/src'); os.environ['SUBSTRATE_CACHE_DIR'] = sys.argv[1] + '/cache'\n"
            "from substrate.gui.app import App\n"
            "app = App(sys.argv[1]); names = [r['name'] for r in app.references_listing()]\n"
            "print(len(names), app.cached_transfer(names) is not None)")
    result = subprocess.run([sys.executable, "-c", code, str(unpacked)], capture_output=True, text=True, check=True)
    count, found = result.stdout.split()
    assert int(count) == summary["references"] and found == "True"


# -- the uploader accepts a static folder ---------------------------------------------------------------------------------------------------
class FakeApi:
    def __init__(self):
        self.calls = []

    def whoami(self):
        return {"name": "Hastika06", "orgs": []}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))

    def update_repo_settings(self, **kw):
        self.calls.append(("update_repo_settings", kw))


@pytest.fixture
def static_folder(tmp_path):
    f = tmp_path / "space"
    (f / "static").mkdir(parents=True)
    (f / "index.html").write_text("<!doctype html><title>x</title>\n", encoding="utf-8")
    (f / "README.md").write_text("---\ntitle: x\nsdk: static\n---\n# x\n", encoding="utf-8")
    (f / "SOURCE_COMMIT").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
    (f / "bundle.zip").write_bytes(b"PK")
    return f


class OldClientApi:
    """huggingface_hub before 0.24 (0.23.4 is installed here): no update_repo_settings, only update_repo_visibility."""

    def __init__(self):
        self.calls = []

    def whoami(self):
        return {"name": "Hastika06", "orgs": []}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))

    def update_repo_visibility(self, **kw):
        self.calls.append(("update_repo_visibility", kw))


def test_public_works_with_a_client_that_only_has_the_older_visibility_call(static_folder, capsys):
    api = OldClientApi()
    assert upload.main([str(static_folder), "--space-id", SPACE, "--public"], api=api) == 0
    assert api.calls[0][2]["private"] is False
    assert api.calls[-1] == ("update_repo_visibility", {"repo_id": SPACE, "private": False, "repo_type": "space"})
    assert "public" in capsys.readouterr().out


def test_a_static_space_is_created_private_as_a_static_space(static_folder, capsys):
    api = FakeApi()
    assert upload.main([str(static_folder), "--space-id", SPACE], api=api) == 0
    assert [c[0] for c in api.calls] == ["create_repo", "upload_folder"]
    create = api.calls[0]
    assert create[2]["space_sdk"] == "static" and create[2]["private"] is True and create[2]["repo_type"] == "space"
    assert "static" in api.calls[1][1]["commit_message"] and "0123456" in api.calls[1][1]["commit_message"]
    assert "private" in capsys.readouterr().out


def test_a_static_space_without_its_page_is_refused(static_folder, capsys):
    (static_folder / "index.html").unlink()
    api = FakeApi()
    assert upload.main([str(static_folder), "--space-id", SPACE], api=api) == 1 and api.calls == []
    assert "no index.html" in capsys.readouterr().err


def test_a_docker_folder_is_still_created_as_a_docker_space(tmp_path):
    f = tmp_path / "space"
    f.mkdir()
    (f / "Dockerfile").write_text("FROM python:3.13-slim-bookworm\n", encoding="utf-8")
    (f / "README.md").write_text("---\ntitle: x\nsdk: docker\napp_port: 7860\n---\n# x\n", encoding="utf-8")
    api = FakeApi()
    assert upload.main([str(f), "--space-id", SPACE], api=api) == 0
    assert api.calls[0][2]["space_sdk"] == "docker"


def test_a_docker_card_with_only_a_page_is_refused(tmp_path, capsys):
    f = tmp_path / "space"
    f.mkdir()
    (f / "index.html").write_text("x", encoding="utf-8")
    (f / "README.md").write_text("---\nsdk: docker\n---\n", encoding="utf-8")
    assert upload.main([str(f), "--space-id", SPACE], api=FakeApi()) == 1
    assert "no Dockerfile" in capsys.readouterr().err
