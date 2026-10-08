"""Source-level security policy: constructs that must not appear in the package, and the few places allowed to start a process or open a socket.

These are static checks over every module in `src/substrate`. They exist so that the claims in SECURITY.md are not a matter of memory: if a new
`subprocess` call, a network client or an unsafe loader appears, this test fails and the change has to be argued for in review (by adding the
module to the allowlist below, with the reason).
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "substrate"
MODULES = sorted(SRC.rglob("*.py"))

FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_MODULES = {"pickle", "cPickle", "marshal", "shelve", "dill", "cloudpickle", "joblib", "ctypes"}
UNSAFE_YAML = {"load", "load_all", "unsafe_load", "unsafe_load_all", "full_load", "full_load_all"}
FORBIDDEN_ATTRIBUTES = {("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execl"), ("os", "spawnl"), ("os", "spawnv")}

#: modules that may start a process, and why
PROCESS_ALLOWED = {
    "qc/__init__.py": "runs the PySCF worker (local or through WSL) with a fixed command, job data on stdin",
}
#: modules that may open or accept a network connection, and why
NETWORK_ALLOWED = {
    "gui/server.py": "the local GUI, bound to 127.0.0.1",
}
NETWORK_MODULES = {"socket", "socketserver", "http", "http.server", "http.client", "urllib", "urllib.request", "urllib3", "requests", "httpx", "aiohttp",
                   "ftplib", "smtplib", "telnetlib", "xmlrpc", "websockets"}


def rel(path):
    return path.relative_to(SRC).as_posix()


def tree_of(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_there_are_modules_to_check():
    assert len(MODULES) > 30


@pytest.mark.parametrize("path", MODULES, ids=rel)
def test_no_dynamic_code_execution_and_no_unsafe_deserialisation(path):
    tree = tree_of(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_CALLS, f"{rel(path)}:{node.lineno} calls {node.func.id}()"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            owner, attribute = node.func.value.id, node.func.attr
            assert (owner, attribute) not in FORBIDDEN_ATTRIBUTES, f"{rel(path)}:{node.lineno} calls {owner}.{attribute}()"
            assert not (owner == "yaml" and attribute in UNSAFE_YAML), f"{rel(path)}:{node.lineno} uses yaml.{attribute}: use safe_load / the safeload module"
    assert not (imported(tree) & FORBIDDEN_MODULES), f"{rel(path)} imports {sorted(imported(tree) & FORBIDDEN_MODULES)}"


@pytest.mark.parametrize("path", MODULES, ids=rel)
def test_no_call_passes_shell_true(path):
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.keyword) and node.arg == "shell":
            assert not (isinstance(node.value, ast.Constant) and node.value.value), f"{rel(path)}:{node.value.lineno} uses shell=True"


@pytest.mark.parametrize("path", MODULES, ids=rel)
def test_only_the_listed_modules_start_processes(path):
    uses = {"subprocess", "multiprocessing", "pty"} & imported(tree_of(path))
    if rel(path) not in PROCESS_ALLOWED:
        assert not uses, f"{rel(path)} imports {sorted(uses)}: add it to PROCESS_ALLOWED with the reason, or do not start a process"


@pytest.mark.parametrize("path", MODULES, ids=rel)
def test_only_the_listed_modules_touch_the_network(path):
    uses = NETWORK_MODULES & imported(tree_of(path))
    if rel(path) not in NETWORK_ALLOWED:
        assert not uses, f"{rel(path)} imports {sorted(uses)}: add it to NETWORK_ALLOWED with the reason, or do not open a connection"


def test_the_allowlists_name_modules_that_exist_and_still_use_what_they_are_allowed():
    for name in (*PROCESS_ALLOWED, *NETWORK_ALLOWED):
        assert (SRC / name).is_file(), f"{name} is on an allowlist but does not exist"
    assert "subprocess" in imported(tree_of(SRC / "qc/__init__.py"))
    assert NETWORK_MODULES & imported(tree_of(SRC / "gui/server.py"))


def test_every_yaml_read_goes_through_the_safe_loader():
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        if rel(path) != "safeload.py":
            assert "yaml.safe_load" not in text, f"{rel(path)} reads YAML directly: use substrate.safeload.load_yaml"
