"""The Hugging Face Space: what the builder ships and what it refuses to ship, and what the uploader checks before it sends anything.

Neither script talks to the network here: the uploader is given a stand-in for Hugging Face's client, and the builder is given a small stand-in for the energy cache.
"""
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "deploy" / "huggingface"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = load("build_space")
upload = load("upload_space")
needs_git = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="the builder ships what git tracks, so it needs a checkout")
SPACE = "Hastika06/substrate-mission-control"
HOST = "hastika06-substrate-mission-control.hf.space"


@pytest.fixture
def cache(tmp_path):
    folder = tmp_path / "cache"
    (folder / "gui").mkdir(parents=True)
    with sqlite3.connect(folder / "qc.sqlite3") as db:
        db.execute("CREATE TABLE energies (key TEXT PRIMARY KEY, payload TEXT)")
        db.executemany("INSERT INTO energies VALUES (?, ?)", [("aa11", json.dumps({"energy": -1.5, "converged": True, "program": "pyscf 2.14.0"})),
                                                              ("bb22", json.dumps({"energy": -2.5, "converged": True, "program": "pyscf 2.14.0"}))])
    (folder / "gui" / "transfer-abc123.json").write_text(json.dumps({"names": ["a", "b"], "matrix": [[0, 1], [1, 0]]}), encoding="utf-8")      # a saved study: stale by design, never shipped
    return folder


# -- the host name -----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("space_id, host", [(SPACE, HOST), ("Some_One/My.Space_v2", "some-one-my-space-v2.hf.space"), ("a/b", "a-b.hf.space")])
def test_the_space_host_is_what_hugging_face_serves_it_from(space_id, host):
    assert build.space_host(space_id) == host


@pytest.mark.parametrize("bad", ["", "nobody", "a/b/c", "/b", "a/", "a b/c", "a/b c", "../x/y", "a/b;rm", "http://x/y", "a/-b"])
def test_a_space_id_must_be_owner_slash_name(bad):
    with pytest.raises(build.BuildError, match="not a Space id"):
        build.space_host(bad)


# -- what the builder ships --------------------------------------------------------------------------------------------------------------------
@needs_git
def test_the_space_holds_the_tracked_sources_the_cache_and_the_two_generated_files(tmp_path, cache):
    out = tmp_path / "space"
    summary = build.build(out, SPACE, cache)
    for needed in ("Dockerfile", "README.md", "SOURCE_COMMIT", "pyproject.toml", "requirements-lock.txt", "src/substrate/__init__.py", "src/substrate/gui/static/index.html",
                   "src/substrate/gui/static/app.js", "experiments/proton_transfer.yaml", "cache/qc.sqlite3", "prewarm.py"):
        assert (out / needed).is_file(), needed
    assert not (out / "cache" / "gui").exists() and not list(out.rglob("transfer-*.json"))           # a saved study is keyed to the bytes of the files it was made from: it is computed in the image
    assert summary["host"] == HOST and summary["energies"] == 2 and "studies" not in summary and summary["references"] >= 20
    with sqlite3.connect(out / "cache" / "qc.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM energies").fetchone()[0] == 2
    assert not (out / "tests").exists() and not (out / "deploy").exists() and not list(out.rglob("__pycache__")) and not list(out.rglob("*.pyc"))


@needs_git
def test_nothing_untracked_is_shipped(tmp_path, cache):
    stray = ROOT / "src" / "substrate" / "stray_untracked_file_for_the_test.py"
    stray.write_text("secret = 1\n", encoding="utf-8")
    try:
        out = tmp_path / "space"
        build.build(out, SPACE, cache)
        assert not (out / "src" / "substrate" / stray.name).exists()
    finally:
        stray.unlink()


@needs_git
def test_the_dockerfile_runs_as_a_user_listens_on_the_space_port_and_answers_to_one_host(tmp_path, cache):
    out = tmp_path / "space"
    build.build(out, SPACE, cache)
    text = (out / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim-bookworm" in text and "useradd --create-home --uid 1000 user" in text and "\nUSER user\n" in text
    assert "EXPOSE 7860" in text and 'CMD ["python", "-m", "substrate", "gui"' in text
    assert text.index("COPY --chown=user cache") < text.index("RUN python prewarm.py") < text.index("EXPOSE 7860")        # the study is computed from the stored energies, then the server starts
    cmd = json.loads(text[text.index("CMD ") + 4:].strip())
    assert cmd[cmd.index("--bind") + 1] == "0.0.0.0" and cmd[cmd.index("--port") + 1] == "7860" and cmd[cmd.index("--max-jobs") + 1] == "2"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--allow-host"] == [HOST]                    # exactly one name, and it is this Space's
    assert "--no-browser" in cmd and cmd[cmd.index("--root") + 1] == "/home/user/app"
    assert "SUBSTRATE_MAX_ENSEMBLE=200" in text and "SUBSTRATE_CACHE_DIR=/home/user/cache" in text
    for forbidden in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ADD http", "curl ", "wget ", "sudo", "--privileged", "chmod 777"):
        assert forbidden.lower() not in text.lower(), forbidden


@needs_git
def test_the_card_is_a_docker_space_with_the_license_left_open_and_says_what_it_cannot_do(tmp_path, cache):
    out = tmp_path / "space"
    build.build(out, SPACE, cache)
    card = (out / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(card.split("---")[1])
    assert front["sdk"] == "docker" and front["app_port"] == 7860 and front["license"] == "other" and len(front["short_description"]) <= 60
    assert "No new quantum chemistry" in card and "PySCF is not installed" in card and "2 energies" in card and "At most 2 jobs" in card
    assert "computed when the Space is built" in card and "saved study" not in card and "downstream rates" in card
    assert "https://github.com/hastikacheddy/substrate" in card and "docs/limits.md" in card and "research prototype" in card
    commit = (out / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    assert commit == "unknown" or len(commit) == 40
    assert commit[:7] in card


@needs_git
def test_without_a_cache_the_space_says_so_instead_of_pretending(tmp_path):
    out = tmp_path / "space"
    summary = build.build(out, SPACE, None)
    assert summary["energies"] == 0 and (out / "cache" / "README.txt").is_file()
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "no stored energies" in card and "computed when the Space is built" not in card
    assert "prewarm" not in (out / "Dockerfile").read_text(encoding="utf-8") and not (out / "prewarm.py").exists()          # there is nothing to compute a study from, and a build that tried would fail


# -- what the builder refuses -----------------------------------------------------------------------------------------------------------------
@needs_git
def test_it_never_writes_into_a_folder_that_has_something_in_it(tmp_path, cache):
    out = tmp_path / "space"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(build.BuildError, match="not empty"):
        build.build(out, SPACE, cache)
    assert [p.name for p in out.iterdir()] == ["keep.txt"] and (out / "keep.txt").read_text(encoding="utf-8") == "mine"


@needs_git
@pytest.mark.parametrize("poison", ["C:\\Users\\someone\\project\\x.xyz", "/home/someone/run", "ran on localhost:8080", "see c:/work/file", "owner is Hasti"])
def test_a_cache_value_that_shows_this_machine_stops_the_build_and_leaves_nothing_behind(tmp_path, cache, poison):
    with sqlite3.connect(cache / "qc.sqlite3") as db:
        db.execute("INSERT INTO energies VALUES (?, ?)", ("cc33", json.dumps({"energy": -3.0, "note": poison})))
    out = tmp_path / "space"
    with pytest.raises(build.BuildError, match="publish something from this machine"):
        build.build(out, SPACE, cache, extra_markers=("hasti",))
    assert not out.exists() or not any(out.iterdir())


@needs_git
def test_a_missing_cache_is_an_error_not_an_empty_space(tmp_path):
    with pytest.raises(build.BuildError, match="no energy cache"):
        build.build(tmp_path / "space", SPACE, tmp_path / "nowhere")


def test_the_builder_scans_for_this_machines_name_and_paths():
    markers = build.machine_markers(("extra",))
    assert Path.home().name.lower() in markers and "c:/" in markers and "localhost" in markers and "extra" in markers


def test_the_command_line_reports_a_refusal_and_exits_nonzero(tmp_path, capsys):
    out = tmp_path / "space"
    out.mkdir()
    (out / "x").write_text("x", encoding="utf-8")
    assert build.main([str(out), "--space-id", SPACE]) == 1
    assert "not empty" in capsys.readouterr().err
    assert build.main([str(tmp_path / "other"), "--space-id", "nonsense"]) == 1


# -- the uploader ---------------------------------------------------------------------------------------------------------------------------------
class FakeApi:
    def __init__(self, name="Hastika06", orgs=(), logged_in=True):
        self.name, self.orgs, self.logged_in, self.calls = name, orgs, logged_in, []

    def whoami(self):
        if not self.logged_in:
            raise RuntimeError("Invalid user token. The token is hf_THISMUSTNEVERBEPRINTED")
        return {"name": self.name, "orgs": [{"name": o} for o in self.orgs]}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))

    def update_repo_settings(self, **kw):
        self.calls.append(("update_repo_settings", kw))


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "space"
    (f / "src").mkdir(parents=True)
    (f / "Dockerfile").write_text("FROM python:3.13-slim-bookworm\n", encoding="utf-8")
    (f / "README.md").write_text("---\ntitle: x\nsdk: docker\napp_port: 7860\n---\n# x\n", encoding="utf-8")
    (f / "SOURCE_COMMIT").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
    (f / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    return f


def test_the_space_is_created_private_and_uploaded(folder, capsys):
    api = FakeApi()
    assert upload.main([str(folder), "--space-id", SPACE], api=api) == 0
    kinds = [c[0] for c in api.calls]
    assert kinds == ["create_repo", "upload_folder"]                                                  # and not flipped public
    create = api.calls[0]
    assert create[1] == SPACE and create[2]["private"] is True and create[2]["repo_type"] == "space" and create[2]["space_sdk"] == "docker"
    sent = api.calls[1][1]
    assert sent["folder_path"] == str(folder) and sent["repo_id"] == SPACE and sent["repo_type"] == "space" and "0123456" in sent["commit_message"]
    assert "private" in capsys.readouterr().out


def test_public_is_a_separate_deliberate_flag(folder):
    api = FakeApi()
    assert upload.main([str(folder), "--space-id", SPACE, "--public"], api=api) == 0
    assert api.calls[0][2]["private"] is False and api.calls[-1] == ("update_repo_settings", {"repo_id": SPACE, "repo_type": "space", "private": False})


def test_not_being_logged_in_says_how_to_log_in_and_never_echoes_what_the_library_said(folder, capsys):
    assert upload.main([str(folder), "--space-id", SPACE], api=FakeApi(logged_in=False)) == 1
    err = capsys.readouterr().err
    assert "hf auth login" in err and "HF_TOKEN" in err and "THISMUSTNEVERBEPRINTED" not in err


class HubRefusal(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.response = type("Response", (), {"status_code": status})()


class RefusingApi(FakeApi):
    def __init__(self, status, message):
        super().__init__()
        self.refusal = HubRefusal(status, message)

    def create_repo(self, repo_id, **kw):
        raise self.refusal


def test_a_payment_required_refusal_is_reported_as_what_it_is_and_nothing_is_uploaded(folder, capsys):
    api = RefusingApi(402, "402 Client Error: Payment Required for url: https://huggingface.co/api/repos/create (Request ID: x)\n\nDocker Spaces on free cpu-basic require a PRO subscription.")
    assert upload.main([str(folder), "--space-id", SPACE], api=api) == 1
    err = capsys.readouterr().err
    assert "402" in err and "PRO subscription" in err and "Nothing was made public" in err and "Traceback" not in err
    assert api.calls == []                                                                              # create_repo failed: no upload, no flip to public


def test_another_refusal_shows_its_kind_but_not_the_hub_s_text(folder, capsys):
    api = RefusingApi(500, "boom\nthe request carried Bearer hf_THISMUSTNEVERBEPRINTED")
    assert upload.main([str(folder), "--space-id", SPACE], api=api) == 1
    err = capsys.readouterr().err
    assert "500" in err and "THISMUSTNEVERBEPRINTED" not in err


def test_a_space_under_somebody_elses_name_is_refused(folder, capsys):
    api = FakeApi()
    assert upload.main([str(folder), "--space-id", "someone-else/x"], api=api) == 1 and api.calls == []
    assert "not you or one of your organizations" in capsys.readouterr().err
    org_api = FakeApi(orgs=("my-lab",))
    assert upload.main([str(folder), "--space-id", "my-lab/x"], api=org_api) == 0                      # an organization you belong to is fine


@pytest.mark.parametrize("name", [".env", ".env.local", "id_rsa", "server.pem", "deploy.key", "credentials.json", "hf_token.txt", "sub/dir/my_token", ".netrc", "bundle.p12"])
def test_a_file_that_looks_like_a_secret_is_never_uploaded(folder, name, capsys):
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")
    api = FakeApi()
    assert upload.main([str(folder), "--space-id", SPACE], api=api) == 1 and api.calls == []
    assert "look like secrets" in capsys.readouterr().err


def test_an_ordinary_file_with_a_similar_looking_path_is_not_mistaken_for_a_secret(folder):
    (folder / "src" / "monkey.py").write_text("x = 1\n", encoding="utf-8")                             # "key" in a name is not a .key file
    (folder / "tokens").mkdir()                                                                         # a directory is not checked, only the files in it
    assert "src/monkey.py" in upload.check_folder(folder)


@pytest.mark.parametrize("remove, message", [("Dockerfile", "no Dockerfile"), ("README.md", "no README.md")])
def test_a_folder_without_its_dockerfile_or_card_is_refused(folder, remove, message, capsys):
    (folder / remove).unlink()
    assert upload.main([str(folder), "--space-id", SPACE], api=FakeApi()) == 1
    assert message in capsys.readouterr().err


def test_a_card_that_is_not_a_docker_space_is_refused(folder, capsys):
    (folder / "README.md").write_text("---\ntitle: x\nsdk: gradio\n---\n", encoding="utf-8")
    assert upload.main([str(folder), "--space-id", SPACE], api=FakeApi()) == 1 and "sdk: docker" in capsys.readouterr().err
    assert upload.main([str(folder / "missing"), "--space-id", SPACE], api=FakeApi()) == 1
