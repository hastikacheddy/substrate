"""The deployment files (Dockerfile, Compose, Kubernetes, Prometheus, Grafana, OpenTelemetry): their structure, the security properties they claim, and that everything
they refer to is real: commands the CLI accepts, metrics and labels the code exposes, files that exist, services that are defined and can reach each other.

None of this starts a container. The last two tests run the real `docker compose config` and `kubectl kustomize` when those programs are installed (both parse files
and need no daemon or cluster); the rest read the YAML directly."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
K8S = DEPLOY / "k8s"
SECRETISH_NAME = re.compile(r"(?i)(password|passwd|secret|token|api_?key|credential)")


def load(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_all(path) -> list[dict]:
    return [d for d in yaml.safe_load_all(Path(path).read_text(encoding="utf-8")) if d]


def platform_parser() -> argparse.ArgumentParser:
    pytest.importorskip("sqlalchemy")
    from substrate.control.cli import add_parser
    parser = argparse.ArgumentParser(prog="substrate", exit_on_error=False)
    add_parser(parser.add_subparsers(dest="command", required=True))
    return parser


@pytest.fixture(scope="module")
def compose() -> dict:
    return load(DEPLOY / "docker-compose.yml")


def internal_networks(compose) -> set[str]:
    return {name for name, config in compose["networks"].items() if (config or {}).get("internal")}


def shared_network(compose, a: str, b: str) -> bool:
    services = compose["services"]
    return bool(set(services[a]["networks"]) & set(services[b]["networks"]))


# -- Dockerfile ---------------------------------------------------------------------------------------------------------------------------
def test_the_image_is_built_from_a_pinned_base_and_runs_as_an_unprivileged_user():
    text = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
    froms = re.findall(r"(?mi)^FROM\s+(\S+)", text)
    assert len(froms) == 2 and all(":" in f and not f.endswith(":latest") for f in froms)        # build stage and final stage, each with an explicit tag
    final = text[text.lower().rindex("\nfrom "):]
    assert re.search(r"(?m)^USER\s+10001:10001\s*$", final) and "HEALTHCHECK" in final
    assert "--no-index" in final                                                                  # the final image installs from the wheels built in the first stage only
    assert not re.search(r"(?mi)^ADD\s+https?://|curl[^\n]*\|\s*(sh|bash)|wget[^\n]*\|\s*(sh|bash)", text)
    assert not re.search(r"(?mi)^(ENV|ARG)\s+\w*(PASSWORD|SECRET|TOKEN|API_?KEY)\w*", text)
    assert 'ENTRYPOINT ["python", "-m", "substrate"]' in text


def test_the_extras_the_image_installs_exist_and_every_package_in_them_is_pinned_in_the_platform_lock():
    tomllib = pytest.importorskip("tomllib")                                                      # Python 3.11 and later
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    named = re.findall(r'\[([a-z0-9,]+)\]"', (DEPLOY / "Dockerfile").read_text(encoding="utf-8"))
    assert named and all(set(n.split(",")) <= set(extras) for n in named)
    lock = (ROOT / "requirements-platform-lock.txt").read_text(encoding="utf-8")
    pinned = {re.split(r"[=;\s]", line)[0].lower().replace("_", "-") for line in lock.splitlines() if re.match(r"[A-Za-z]", line)}
    wanted = {re.split(r"[<>=!~\[ ]", d)[0].lower().replace("_", "-") for n in {e for group in named for e in group.split(",")} for d in extras[n]}
    assert wanted <= pinned, sorted(wanted - pinned)


def test_the_platform_lock_is_versions_only_and_nothing_is_pinned_twice():
    names = []
    for line in (ROOT / "requirements-platform-lock.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("-r "):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([0-9][A-Za-z0-9_.\-]*)(\s*;.*)?", line)
        assert match, f"not a plain pin: {line}"
        names.append(match.group(1).lower().replace("_", "-"))
    assert len(names) == len(set(names))


def test_the_build_context_leaves_out_secrets_and_history():
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").split())
    assert {".git", "deploy/secrets", ".env"} <= ignored
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "deploy/secrets/*" in gitignore and "!deploy/secrets/README.md" in gitignore


def publishable_files() -> list[Path]:
    """The files under deploy/ that git would commit: tracked ones plus untracked ones that are not ignored (a developer's own secret files are ignored, so they are not these).
    Outside a checkout, every file."""
    try:
        listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "deploy"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if listing.returncode == 0 and listing.stdout:
            return [ROOT / name for name in listing.stdout.split("\x00") if name and (ROOT / name).is_file()]
    except (OSError, subprocess.SubprocessError):
        pass
    return [p for p in DEPLOY.rglob("*") if p.is_file()]


def test_no_secret_is_committed_under_deploy():
    files = publishable_files()
    assert sorted(p.name for p in files if p.parent == DEPLOY / "secrets") == ["README.md"]
    # `existingSecret: <name>` (the chart's value) names a Kubernetes Secret object; it holds no secret, so it is the one key that is not a credential.
    pattern = re.compile(r"sbs_[A-Za-z0-9]{6,}_|-----BEGIN [A-Z ]*PRIVATE KEY|(?<!existing)(?<!Existing)(?i:password|secret|token)\s*[:=]\s*[\"']?[A-Za-z0-9+_\-]{16,}")
    for path in files:
        if path.suffix in (".yml", ".yaml", ".json", ".md", "") and path.name != "Dockerfile":
            for line in path.read_text(encoding="utf-8").splitlines():
                assert not pattern.search(line), f"{path.relative_to(ROOT)}: {line.strip()[:80]}"


# -- Compose ------------------------------------------------------------------------------------------------------------------------------
def test_every_container_is_locked_down(compose):
    assert set(compose["services"]) == {"db", "redis", "api", "worker", "otel-collector", "tempo", "prometheus", "grafana"}
    for name, service in compose["services"].items():
        assert service["read_only"] is True, name
        assert service["cap_drop"] == ["ALL"] and "cap_add" not in service, name
        assert "no-new-privileges:true" in service["security_opt"], name
        assert str(service["user"]).split(":")[0] not in ("", "0", "root"), name
        assert service.get("pids_limit"), name
        for forbidden in ("privileged", "network_mode", "pid", "ipc", "devices", "userns_mode", "cgroup_parent", "sysctls", "extra_hosts"):
            assert forbidden not in service, (name, forbidden)
        for volume in service.get("volumes", []):
            assert "docker.sock" not in volume and volume.split(":")[0] not in ("/", "/var", "/etc", "/proc", "/sys", "/dev"), (name, volume)


def test_images_are_pinned_to_a_tag_and_the_substrate_image_is_built_from_the_repository(compose):
    for name, service in compose["services"].items():
        image = service["image"]
        if name in ("api", "worker"):
            assert image.startswith("substrate:") and service["build"] == {"context": "..", "dockerfile": "deploy/Dockerfile"}
        else:
            assert re.search(r":v?[0-9][\w.\-]*$", image), (name, image)             # a version, not latest and not absent
    assert (DEPLOY / "Dockerfile").exists() and (ROOT / "pyproject.toml").exists()


def test_ports_are_published_on_loopback_only_and_only_by_services_that_can_be_reached(compose):
    internal = internal_networks(compose)
    published = {n: s["ports"] for n, s in compose["services"].items() if s.get("ports")}
    assert set(published) == {"api", "grafana"}
    for name, ports in published.items():
        assert all(str(p).startswith("127.0.0.1:") for p in ports), name
        assert set(compose["services"][name]["networks"]) - internal, f"{name} publishes a port but is only on internal networks (it would not work)"


def test_the_network_verifier_names_exactly_the_containers_with_missing_or_extra_networks(compose):
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_networks", DEPLOY / "verify_networks.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    config = {"name": "substrate", "services": {n: {"networks": {x: None for x in s["networks"]}} for n, s in compose["services"].items()},
              "networks": {k: {"name": f"substrate_{k}"} for k in compose["networks"]}}
    wanted = verifier.expected_networks(config, "substrate")
    assert wanted["api"] == {"substrate_frontend", "substrate_backend", "substrate_monitoring"} and wanted["db"] == {"substrate_backend"}

    def container(service, number=1):
        return {"Name": f"/substrate-{service}-{number}", "Config": {"Labels": {"com.docker.compose.service": service}},
                "NetworkSettings": {"Networks": {n: {} for n in wanted[service]}}}
    assert verifier.mismatches(wanted, [container(s) for s in wanted]) == []
    lost, gained = container("api"), container("worker", 2)
    del lost["NetworkSettings"]["Networks"]["substrate_frontend"]
    gained["NetworkSettings"]["Networks"]["substrate_frontend"] = {}
    assert verifier.mismatches(wanted, [container("db"), lost, gained]) == ["substrate-api-1 (api): missing substrate_frontend",
                                                                          "substrate-worker-2 (worker): has unexpected substrate_frontend"]


def test_the_database_and_the_workers_have_no_route_out_and_the_workers_reach_nothing_else(compose):
    internal = internal_networks(compose)
    services = compose["services"]
    assert set(services["db"]["networks"]) <= internal and set(services["worker"]["networks"]) <= internal
    assert set(services["worker"]["networks"]) == {"backend", "telemetry"}
    peers = {name for name, s in services.items() if name != "worker" and set(s["networks"]) & set(services["worker"]["networks"])}
    assert peers == {"db", "redis", "api", "otel-collector"}                                  # the database, Redis and the API (backend), and the collector (telemetry)
    assert "frontend" not in services["worker"]["networks"] and "frontend" not in services["db"]["networks"]
    assert set(compose["networks"]) == {"frontend", "backend", "telemetry", "monitoring"} and {"backend", "telemetry", "monitoring"} <= internal
    for name, service in services.items():
        assert set(service["networks"]) <= set(compose["networks"]), name
        for dependency in (service.get("depends_on") or {}):
            assert dependency in services, (name, dependency)


def test_secrets_are_files_never_environment_values_and_each_is_mounted_where_it_is_named(compose):
    top = compose["secrets"]
    assert all(set(v) == {"file"} and v["file"].startswith("./secrets/") for v in top.values())
    used = set()
    for name, service in compose["services"].items():
        mounted = set(service.get("secrets", []))
        assert mounted <= set(top), name
        used |= mounted
        for key, value in (service.get("environment") or {}).items():
            value = str(value)
            if SECRETISH_NAME.search(key):
                assert key.endswith("_FILE") or key.endswith("__FILE"), (name, key)           # the value is a path, not the credential
            assert not re.search(r"sbs_[A-Za-z0-9]{6,}_|://[^/\s:@]+:[^/\s@]+@", value), (name, key)
            match = re.fullmatch(r"/run/secrets/(\w+)", value)
            if match:
                assert match.group(1) in mounted, (name, key, "names a secret the service does not have")
    assert used == set(top)


def test_bind_mounts_exist_are_read_only_and_named_volumes_are_defined(compose):
    for name, service in compose["services"].items():
        for volume in service.get("volumes", []):
            source, target, *mode = volume.split(":")
            if source.startswith("./"):
                assert (DEPLOY / source).exists(), (name, source)
                assert mode == ["ro"], (name, volume)
            else:
                assert source in compose["volumes"], (name, source)
    for name in ("db", "worker", "api"):
        assert "secrets" not in compose["services"][name].get("volumes", [])


def test_the_api_and_worker_commands_are_ones_the_cli_accepts(compose):
    parser = platform_parser()
    api = parser.parse_args(compose["services"]["api"]["command"])
    assert (api.platform_command, api.host, api.port, api.allow_external) == ("serve", "0.0.0.0", 8080, True)
    worker = parser.parse_args(compose["services"]["worker"]["command"])
    assert (worker.platform_command, worker.queues, worker.isolation) == ("worker", "cpu,quantum", "process")
    assert compose["services"]["api"]["ports"] == ["127.0.0.1:${SUBSTRATE_API_PORT:-8080}:8080"]
    for name in ("api", "worker"):
        env = compose["services"][name]["environment"]
        assert env["SUBSTRATE_DB_FILE"] == "/run/secrets/db_url" and "db_url" in compose["services"][name]["secrets"]
        assert env["SUBSTRATE_STORE"] == "/data/artifacts"


def test_the_api_healthcheck_asks_a_route_the_api_has(compose):
    text = " ".join(compose["services"]["api"]["healthcheck"]["test"])
    port, path = re.search(r"127\.0\.0\.1:(\d+)(/\w+)", text).groups()
    assert port == "8080" and f'"{path}"' in (ROOT / "src/substrate/control/api.py").read_text(encoding="utf-8")
    assert re.search(r'HEALTHCHECK[^\n]*\n?[^\n]*127\.0\.0\.1:8080/healthz', (DEPLOY / "Dockerfile").read_text(encoding="utf-8"))


def test_the_workers_send_traces_where_the_collector_listens_and_the_collector_hands_them_to_tempo(compose):
    collector = load(DEPLOY / "otel/collector.yaml")
    receiver_port = collector["receivers"]["otlp"]["protocols"]["http"]["endpoint"].rsplit(":", 1)[1]
    assert compose["services"]["worker"]["environment"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://otel-collector:{receiver_port}"
    assert shared_network(compose, "worker", "otel-collector")
    pipeline = collector["service"]["pipelines"]["traces"]
    for kind in ("receivers", "processors", "exporters"):
        assert all(name in collector[kind] for name in pipeline[kind]), kind                # every component the pipeline names is configured
    assert set(collector["service"]["extensions"]) <= set(collector["extensions"])
    exporter = collector["exporters"]["otlphttp/tempo"]["endpoint"]
    tempo = load(DEPLOY / "tempo/tempo.yaml")
    assert exporter == f"http://tempo:{tempo['distributor']['receivers']['otlp']['protocols']['http']['endpoint'].rsplit(':', 1)[1]}"
    assert shared_network(compose, "otel-collector", "tempo")
    assert not (set(compose["services"]["tempo"]["networks"]) & set(compose["services"]["worker"]["networks"]))      # a worker cannot reach the trace store directly


def test_prometheus_scrapes_the_api_with_a_key_it_is_given_as_a_secret_file(compose):
    config = load(DEPLOY / "prometheus/prometheus.yml")
    (job,) = config["scrape_configs"]
    (target,) = job["static_configs"][0]["targets"]
    assert target == "api:8080" and shared_network(compose, "prometheus", "api")
    assert f'"{job["metrics_path"]}"' in (ROOT / "src/substrate/control/api.py").read_text(encoding="utf-8")
    credentials = job["authorization"]["credentials_file"]
    assert credentials == "/run/secrets/prometheus_token" and "prometheus_token" in compose["services"]["prometheus"]["secrets"]
    assert job["authorization"]["type"] == "Bearer"
    mounts = {v.split(":")[1] for v in compose["services"]["prometheus"]["volumes"]}
    assert {config["rule_files"][0], "/etc/prometheus/prometheus.yml"} <= mounts


def test_grafana_provisioning_points_at_services_that_exist_and_at_a_dashboard_directory_it_mounts(compose):
    sources = {d["uid"]: d for d in load(DEPLOY / "grafana/provisioning/datasources/datasources.yml")["datasources"]}
    assert set(sources) == {"prometheus", "tempo"}
    tempo = load(DEPLOY / "tempo/tempo.yaml")
    assert sources["prometheus"]["url"] == "http://prometheus:9090" and sources["tempo"]["url"] == f"http://tempo:{tempo['server']['http_listen_port']}"
    assert shared_network(compose, "grafana", "prometheus") and shared_network(compose, "grafana", "tempo")
    (provider,) = load(DEPLOY / "grafana/provisioning/dashboards/dashboards.yml")["providers"]
    mounts = {v.split(":")[1]: v.split(":")[0] for v in compose["services"]["grafana"]["volumes"] if v.startswith("./")}
    assert provider["options"]["path"] in mounts and list((DEPLOY / mounts[provider["options"]["path"]]).glob("*.json"))
    assert "/etc/grafana/provisioning" in mounts


# -- the metrics the dashboards and alerts name -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def exposed(tmp_path_factory) -> dict[str, set[str]]:
    """Every sample name /metrics produces after a run that succeeded, a run that was stopped by a limit, one still waiting, a refusal and a failed login,
    with the label names each carries: so a dashboard cannot name a series, or a label of one, that the code does not emit."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("prometheus_client")
    pytest.importorskip("opentelemetry.sdk")
    from conftest import EXPERIMENTS
    from substrate.control.artifacts import Artifacts, LocalArtifactStore
    from substrate.control.observability import MetricsObserver, with_live_gauges
    from substrate.control.orchestrator import Orchestrator
    from substrate.control.registry import Registry
    from substrate.control.security import Role
    from substrate.control.worker import Worker
    from substrate.experiment import read_spec
    tmp = tmp_path_factory.mktemp("exposed")
    registry = Registry(f"sqlite:///{tmp / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 50)
    who = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    metrics = with_live_gauges(MetricsObserver(), registry)
    orchestrator = Orchestrator(registry, Artifacts(registry, LocalArtifactStore(tmp / "s")), observer=metrics)
    experiment = registry.add_experiment(who, "x", read_spec(EXPERIMENTS / "proton_transfer.yaml"))["id"]
    orchestrator.submit(who, experiment, n_samples=0)
    Worker(orchestrator).run_until_idle()
    patch = pytest.MonkeyPatch()
    patch.setenv("SUBSTRATE_MAX_GRID_POINTS", "10")
    try:
        orchestrator.submit(who, experiment, n_samples=0)
        Worker(orchestrator).run_until_idle()
    finally:
        patch.undo()
    orchestrator.submit(who, experiment, n_samples=0)                                         # left waiting: queue depth and active runs have a value
    metrics.auth_failed("bad-token")
    names: dict[str, set[str]] = {}
    for family in metrics.registry.collect():
        for sample in family.samples:
            names.setdefault(sample.name, set()).update(sample.labels)
    names["up"] = {"job", "instance", "service"}                                              # Prometheus adds this one itself
    return names


SELECTOR = re.compile(r"\b(substrate_[a-z_]+|up)\s*\{([^}]*)\}")
LABEL = re.compile(r'(\w+)\s*(=~|!~|!=|=)\s*"([^"]*)"')
GROUPING = re.compile(r"\b(?:by|without)\s*\(([^)]*)\)")
METRIC = re.compile(r"\b(substrate_[a-z_]+)\b")


def domains() -> dict[tuple[str, str], set[str]]:
    from substrate.control.status import AXES, State
    runs = {"queued", "running", "succeeded", "failed", "cancelled"}
    tasks = {"pending", "ready", "running", "succeeded", "failed", "skipped", "cancelled"}
    return {("substrate_run_axis_recent", "axis"): set(AXES), ("substrate_run_axis_recent", "state"): {s.value for s in State},
            ("substrate_runs_recent", "state"): runs, ("substrate_runs_active", "state"): {"queued", "running"}, ("substrate_tasks_recent", "state"): tasks,
            ("substrate_task_failures_recent", "axis"): set(AXES), ("substrate_task_seconds_recent", "stat"): {"mean", "max"}}


def balanced(expr: str) -> bool:
    stack, closing = [], {")": "(", "]": "[", "}": "{"}
    in_string = False
    for ch in expr:
        if ch == '"':
            in_string = not in_string
        elif not in_string and ch in "([{":
            stack.append(ch)
        elif not in_string and ch in closing:
            if not stack or stack.pop() != closing[ch]:
                return False
    return not stack and not in_string


def check_expression(expr: str, exposed: dict[str, set[str]], legend: str = "") -> None:
    """Not promtool: it cannot tell whether a PromQL expression is well typed. It checks that the brackets balance, that every substrate_* series is one the code
    emits, that every label in a selector, a `by` clause or a legend belongs to those series, and that the literal values of the state-like labels are real."""
    assert balanced(expr), expr
    used = set(METRIC.findall(expr)) | ({"up"} if re.search(r"\bup\s*\{", expr) else set())
    assert used, expr
    for metric in used:
        assert metric in exposed, f"{metric} is not emitted ({expr})"
    union = set().union(*(exposed[m] for m in used))
    domain = domains()
    for metric, body in SELECTOR.findall(expr):
        for label, operator, value in LABEL.findall(body):
            assert label in exposed[metric], f"{metric} has no label {label}"
            if operator == "=" and (metric, label) in domain:
                assert value in domain[(metric, label)], f"{metric}{{{label}={value!r}}} is not a value it takes"
    grouped = set()
    for clause in GROUPING.findall(expr):
        labels = {x.strip() for x in clause.split(",") if x.strip()}
        assert labels <= union, f"grouping by {labels - union} which no series in the expression has"
        grouped |= labels
    for label in re.findall(r"\{\{(\w+)\}\}", legend):
        assert label in (grouped or union), f"legend names {label}, which the expression does not keep"


def test_every_series_a_dashboard_panel_draws_is_one_the_code_emits(exposed):
    dashboard = json.loads((DEPLOY / "grafana/dashboards/substrate.json").read_text(encoding="utf-8"))
    sources = {d["uid"] for d in load(DEPLOY / "grafana/provisioning/datasources/datasources.yml")["datasources"]}
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 12 and dashboard["uid"] and dashboard["editable"] is False
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] in sources and panel["title"] and panel["description"], panel["title"]
        grid = panel["gridPos"]
        assert 0 <= grid["x"] and grid["x"] + grid["w"] <= 24 and grid["w"] > 0 and grid["h"] > 0, panel["title"]
        assert panel["targets"], panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] in sources
            check_expression(target["expr"], exposed, target.get("legendFormat", ""))
    boxes = [(p["gridPos"]["x"], p["gridPos"]["y"], p["gridPos"]["w"], p["gridPos"]["h"]) for p in dashboard["panels"]]
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        for x2, y2, w2, h2 in boxes[i + 1:]:
            assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1, "two panels overlap"


def test_every_alert_names_series_the_code_emits_and_says_what_it_means(exposed):
    groups = load(DEPLOY / "prometheus/alerts.yml")["groups"]
    rules = [r for g in groups for r in g["rules"]]
    assert len(rules) >= 6 and len({r["alert"] for r in rules}) == len(rules)
    for rule in rules:
        check_expression(rule["expr"], exposed)
        assert rule["labels"]["severity"] in ("warning", "critical") and rule["annotations"]["summary"], rule["alert"]
        assert re.fullmatch(r"\d+[smh]", str(rule["for"])), rule["alert"]


def test_the_live_series_the_dashboards_rely_on_are_all_emitted(exposed):
    for name in ("substrate_queue_depth", "substrate_runs_active", "substrate_build_info", "substrate_tasks_recent", "substrate_task_failures_recent", "substrate_task_seconds_recent",
                 "substrate_runs_recent", "substrate_run_axis_recent", "substrate_runs_submitted_total", "substrate_blocked_total", "substrate_auth_failures_total"):
        assert name in exposed, name
    assert exposed["substrate_run_axis_recent"] == {"axis", "state"} and exposed["substrate_task_seconds_recent"] == {"kind", "stat"}


def test_the_expression_checker_catches_what_it_is_for(exposed):
    for bad in ('sum(substrate_no_such_series)', 'substrate_runs_recent{colour="red"}', 'substrate_runs_recent{state="exploded"}', 'sum by (axis) (substrate_runs_recent)',
                'sum(substrate_runs_recent', 'substrate_run_axis_recent{axis="gravity"}'):
        with pytest.raises(AssertionError):
            check_expression(bad, exposed)
    with pytest.raises(AssertionError):
        check_expression("sum by (state) (substrate_runs_recent)", exposed, "{{axis}}")
    check_expression('sum by (state) (substrate_runs_recent{state!="failed"})', exposed, "{{state}}")


# -- Kubernetes ---------------------------------------------------------------------------------------------------------------------------
HELM = DEPLOY / "helm" / "substrate"


@pytest.fixture(scope="module")
def kustomize_manifests() -> dict[str, list[dict]]:
    kustomization = load(K8S / "kustomization.yaml")
    return {name: load_all(K8S / name) for name in kustomization["resources"]}


def find_helm() -> str | None:
    found = shutil.which("helm")
    if found:
        return found
    winget = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "helm.exe"        # where winget puts it, before a new shell has it on its path
    return str(winget) if winget.exists() else None


def helm_render(*settings: str) -> list[dict]:
    """The chart as `helm template` renders it, with `--set` overrides: what a cluster would be asked to create."""
    helm = find_helm()
    if helm is None:
        pytest.skip("helm is not installed")
    command = [helm, "template", "substrate", str(HELM), "--namespace", "substrate"]
    for setting in settings:
        command += ["--set", setting]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    return [d for d in yaml.safe_load_all(result.stdout) if d]


@pytest.fixture(scope="module", params=["kustomize", "helm"])
def manifests(request, kustomize_manifests) -> dict[str, list[dict]]:
    """The same properties are asserted of the plain manifests and of what the chart renders by default."""
    return kustomize_manifests if request.param == "kustomize" else {"helm": helm_render()}


def objects(manifests, kind) -> list[dict]:
    return [o for docs in manifests.values() for o in docs if o["kind"] == kind]


def workloads(manifests) -> list[dict]:
    return objects(manifests, "Deployment") + objects(manifests, "StatefulSet")


def test_the_kustomization_lists_every_manifest_and_each_object_is_in_the_namespace(kustomize_manifests):
    manifests = kustomize_manifests
    on_disk = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert set(manifests) == on_disk
    for docs in manifests.values():
        for o in docs:
            assert o["apiVersion"] and o["metadata"]["name"]
            if o["kind"] != "Namespace":
                assert o["metadata"]["namespace"] == "substrate", o["metadata"]["name"]
    assert load(K8S / "kustomization.yaml")["namespace"] == "substrate"
    (namespace,) = objects(manifests, "Namespace")
    assert namespace["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"
    assert [i["name"] for i in load(K8S / "kustomization.yaml")["images"]] == ["substrate"]


def test_every_pod_meets_the_restricted_standard_and_none_is_given_a_token_or_a_host_resource(manifests):
    found = {w["metadata"]["name"] for w in workloads(manifests)}
    assert found == {"postgres", "substrate-api", "substrate-worker"}
    for workload in workloads(manifests):
        pod = workload["spec"]["template"]["spec"]
        name = workload["metadata"]["name"]
        assert pod["automountServiceAccountToken"] is False and pod["serviceAccountName"] == "substrate", name
        context = pod["securityContext"]
        assert context["runAsNonRoot"] is True and context["runAsUser"] not in (0, None) and context["seccompProfile"] == {"type": "RuntimeDefault"}, name
        for forbidden in ("hostNetwork", "hostPID", "hostIPC", "hostPorts", "shareProcessNamespace"):
            assert not pod.get(forbidden), (name, forbidden)
        assert all("hostPath" not in v for v in pod["volumes"]), name
        for container in pod["containers"]:
            sc = container["securityContext"]
            assert sc["allowPrivilegeEscalation"] is False and sc["readOnlyRootFilesystem"] is True and sc["capabilities"] == {"drop": ["ALL"]}, name
            assert not sc.get("privileged") and "add" not in sc["capabilities"], name
            assert container["resources"]["requests"] and container["resources"]["limits"]["memory"], name
            assert not container["image"].endswith(":latest"), name
            for env in container.get("env", []):
                if SECRETISH_NAME.search(env["name"]):
                    assert env["name"].endswith("_FILE") and env["value"].startswith("/run/secrets/"), (name, env["name"])
                assert "valueFrom" not in env or "secretKeyRef" not in env["valueFrom"], (name, "a secret as an environment value")
    for name in ("substrate-api", "substrate-worker"):
        (w,) = [w for w in workloads(manifests) if w["metadata"]["name"] == name]
        assert w["spec"]["template"]["spec"]["securityContext"]["runAsUser"] == 10001


def test_secrets_are_mounted_as_files_one_item_at_a_time_and_the_paths_agree_with_the_environment(manifests):
    for workload in workloads(manifests):
        pod = workload["spec"]["template"]["spec"]
        name = workload["metadata"]["name"]
        (container,) = pod["containers"]
        mount_of = {m["name"]: m for m in container["volumeMounts"]}
        for volume in pod["volumes"]:
            if "secret" in volume:
                secret = volume["secret"]
                assert secret["secretName"] == "substrate-secrets" and secret["items"] and secret["defaultMode"] <= 0o440, name
                assert mount_of[volume["name"]]["readOnly"] is True
                mounted = {f"{mount_of[volume['name']]['mountPath']}/{item['path']}" for item in secret["items"]}
                for env in container.get("env", []):
                    if env["name"].endswith("_FILE") and env["value"].startswith("/run/secrets/"):
                        assert env["value"] in mounted, (name, env["name"])
        from_templates = {t["metadata"]["name"] for t in workload["spec"].get("volumeClaimTemplates", [])}
        assert {v["name"] for v in pod["volumes"]} | from_templates >= set(mount_of), name
    wanted = {w["metadata"]["name"]: [i["key"] for v in w["spec"]["template"]["spec"]["volumes"] if "secret" in v for i in v["secret"]["items"]] for w in workloads(manifests)}
    assert wanted == {"postgres": ["db_password"], "substrate-api": ["db_url"], "substrate-worker": ["db_url"]}               # each gets only what it needs


def test_services_select_pods_that_exist_and_probes_use_ports_the_pods_declare(manifests):
    pods = {w["metadata"]["name"]: w["spec"]["template"] for w in workloads(manifests)}
    for service in objects(manifests, "Service"):
        selector = service["spec"]["selector"]
        (match,) = [t for t in pods.values() if selector.items() <= t["metadata"]["labels"].items()]
        declared = {p["name"] for c in match["spec"]["containers"] for p in c.get("ports", [])}
        for port in service["spec"]["ports"]:
            assert port["targetPort"] in declared, service["metadata"]["name"]
    api = pods["substrate-api"]["spec"]["containers"][0]
    assert api["readinessProbe"]["httpGet"] == {"path": "/readyz", "port": "http"} and api["livenessProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}
    assert api["ports"] == [{"name": "http", "containerPort": 8080}]
    (statefulset,) = objects(manifests, "StatefulSet")
    (headless,) = [s for s in objects(manifests, "Service") if s["metadata"]["name"] == statefulset["spec"]["serviceName"]]
    assert headless["spec"]["clusterIP"] == "None"
    claims = {o["metadata"]["name"] for o in objects(manifests, "PersistentVolumeClaim")}
    for workload in workloads(manifests):
        for volume in workload["spec"]["template"]["spec"]["volumes"]:
            if "persistentVolumeClaim" in volume:
                assert volume["persistentVolumeClaim"]["claimName"] in claims
    assert objects(manifests, "PersistentVolumeClaim")[0]["spec"]["accessModes"] == ["ReadWriteMany"]
    (budget,) = objects(manifests, "PodDisruptionBudget")
    assert budget["spec"]["selector"]["matchLabels"].items() <= pods["substrate-api"]["metadata"]["labels"].items()


def test_the_container_arguments_are_ones_the_cli_accepts(manifests):
    parser = platform_parser()
    pods = {w["metadata"]["name"]: w["spec"]["template"]["spec"]["containers"][0] for w in workloads(manifests)}
    api = parser.parse_args(pods["substrate-api"]["args"])
    assert (api.platform_command, api.host, api.port, api.allow_external) == ("serve", "0.0.0.0", 8080, True)
    worker = parser.parse_args(pods["substrate-worker"]["args"])
    assert (worker.platform_command, worker.queues, worker.isolation) == ("worker", "cpu,quantum", "process")
    for name in ("substrate-api", "substrate-worker"):
        env = {e["name"]: e["value"] for e in pods[name]["env"]}
        assert env == {"SUBSTRATE_DB_FILE": "/run/secrets/db_url", "SUBSTRATE_STORE": "/data/artifacts"}


def test_the_network_is_closed_by_default_and_opened_only_where_the_design_says(manifests):
    policies = {p["metadata"]["name"]: p["spec"] for p in objects(manifests, "NetworkPolicy")}
    assert policies["default-deny"] == {"podSelector": {}, "policyTypes": ["Ingress", "Egress"]}
    for name, spec in policies.items():
        if name in ("default-deny",):
            continue
        assert spec["podSelector"] == {} or "matchLabels" in spec["podSelector"], name
    # every workload is selected by a policy that gives it exactly the access it needs
    labels = {w["metadata"]["name"]: w["spec"]["template"]["metadata"]["labels"] for w in workloads(manifests)}
    def selected(workload, policy):
        match = policies[policy]["podSelector"].get("matchLabels", {})
        return match.items() <= labels[workload].items()
    assert selected("substrate-api", "api-ingress") and selected("substrate-api", "api-egress") and selected("substrate-worker", "worker-egress") and selected("postgres", "postgres-ingress")
    assert not selected("substrate-worker", "api-egress") and not selected("postgres", "api-egress") and not any(selected("substrate-worker", p) for p in ("api-ingress", "postgres-ingress"))
    # the workers can reach the database and nothing else (DNS aside); the database accepts the API and the workers on its port and nothing else
    (worker_rule,) = policies["worker-egress"]["egress"]
    assert worker_rule["ports"] == [{"protocol": "TCP", "port": 5432}] and "ipBlock" not in str(worker_rule)
    assert worker_rule["to"] == [{"podSelector": {"matchLabels": {"app.kubernetes.io/name": "postgres"}}}]
    (db_rule,) = policies["postgres-ingress"]["ingress"]
    assert {kv for f in db_rule["from"] for kv in f["podSelector"]["matchLabels"].items()} == {("app.kubernetes.io/name", "substrate-api"), ("app.kubernetes.io/name", "substrate-worker")}
    assert db_rule["ports"] == [{"protocol": "TCP", "port": 5432}]
    assert policies["postgres-ingress"]["policyTypes"] == ["Ingress"]
    # the API's way out to the internet excludes private ranges, link-local (the cloud metadata service) and loopback; nothing else is allowed to reach the internet
    external = [rule for rule in policies["api-egress"]["egress"] if "ipBlock" in str(rule)]
    (rule,) = external
    block = rule["to"][0]["ipBlock"]
    assert block["cidr"] == "0.0.0.0/0" and {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "127.0.0.0/8"} <= set(block["except"]) and rule["ports"] == [{"protocol": "TCP", "port": 443}]
    for name, spec in policies.items():
        if name != "api-egress":
            assert "ipBlock" not in str(spec), name
    dns = policies["allow-dns"]["egress"][0]
    assert dns["to"][0]["namespaceSelector"]["matchLabels"] == {"kubernetes.io/metadata.name": "kube-system"} and {p["port"] for p in dns["ports"]} == {53}


# -- the Helm chart ------------------------------------------------------------------------------------------------------------------------
def test_helm_lints_the_chart_without_a_warning_that_matters():
    helm = find_helm()
    if helm is None:
        pytest.skip("helm is not installed")
    result = subprocess.run([helm, "lint", str(HELM), "--strict"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def test_the_chart_can_leave_out_the_bundled_database_and_every_rule_that_names_it():
    docs = helm_render("postgres.enabled=false")
    assert not [d for d in docs if d["kind"] == "StatefulSet"]
    policies = {d["metadata"]["name"]: d["spec"] for d in docs if d["kind"] == "NetworkPolicy"}
    assert "postgres-ingress" not in policies and "postgres" not in json.dumps(policies)
    assert "egress" not in policies["worker-egress"]                                                # a worker with no database in the cluster may reach nothing (the DNS rule is separate)
    assert [r["ports"] for r in policies["api-egress"]["egress"]] == [[{"protocol": "TCP", "port": 443}]]


def test_an_image_digest_replaces_the_tag_and_the_numbers_of_replicas_and_queues_follow_the_values():
    digest = "sha256:" + "ab" * 32
    docs = helm_render(f"image.digest={digest}", "api.replicas=3", "worker.replicas=5", "worker.queues=cpu", "image.repository=registry.example/substrate")
    deployments = {d["metadata"]["name"]: d for d in docs if d["kind"] == "Deployment"}
    for name, deployment in deployments.items():
        assert deployment["spec"]["template"]["spec"]["containers"][0]["image"] == f"registry.example/substrate@{digest}", name
    assert deployments["substrate-api"]["spec"]["replicas"] == 3 and deployments["substrate-worker"]["spec"]["replicas"] == 5
    assert deployments["substrate-worker"]["spec"]["template"]["spec"]["containers"][0]["args"][3] == "cpu"


def test_the_network_policies_can_be_switched_off_and_nothing_else_changes():
    default = helm_render()
    off = helm_render("networkPolicy.enabled=false")
    assert not [d for d in off if d["kind"] == "NetworkPolicy"] and [d for d in default if d["kind"] == "NetworkPolicy"]
    assert [d for d in default if d["kind"] != "NetworkPolicy"] == [d for d in off if d["kind"] != "NetworkPolicy"]


def test_the_chart_and_the_plain_manifests_describe_the_same_workloads(kustomize_manifests):
    helm = {d["metadata"]["name"]: d for d in helm_render() if d["kind"] in ("Deployment", "StatefulSet")}
    plain = {o["metadata"]["name"]: o for docs in kustomize_manifests.values() for o in docs if o["kind"] in ("Deployment", "StatefulSet")}
    assert set(helm) == set(plain)
    for name in plain:
        a, b = helm[name]["spec"]["template"]["spec"], plain[name]["spec"]["template"]["spec"]
        assert a["securityContext"] == b["securityContext"] and a["containers"][0]["securityContext"] == b["containers"][0]["securityContext"], name
        assert a["containers"][0].get("args") == b["containers"][0].get("args") and a["containers"][0]["resources"] == b["containers"][0]["resources"], name
        assert [v["name"] for v in a["volumes"]] == [v["name"] for v in b["volumes"]], name


# -- the real tools, when they are installed ----------------------------------------------------------------------------------------------
def test_docker_compose_itself_accepts_the_compose_file(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    if subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=60).returncode != 0:
        pytest.skip("the docker compose plugin is not installed")
    copy = tmp_path / "deploy"
    shutil.copytree(DEPLOY, copy, ignore=shutil.ignore_patterns("k8s"))
    for secret in ("db_url", "db_password", "prometheus_token", "grafana_admin_password"):
        (copy / "secrets" / secret).write_text("placeholder\n", encoding="utf-8")
    result = subprocess.run(["docker", "compose", "-f", str(copy / "docker-compose.yml"), "config", "--format", "json"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    rendered = json.loads(result.stdout)
    assert set(rendered["services"]) == {"db", "redis", "api", "worker", "otel-collector", "tempo", "prometheus", "grafana"}
    assert rendered["services"]["worker"]["environment"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://otel-collector:4318"      # the merge of the shared block and the worker's own
    assert all(s["read_only"] is True and s["cap_drop"] == ["ALL"] for s in rendered["services"].values())


def test_kubectl_itself_renders_the_kustomization(kustomize_manifests):
    manifests = kustomize_manifests
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is not installed")
    result = subprocess.run(["kubectl", "kustomize", str(K8S)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    rendered = [d for d in yaml.safe_load_all(result.stdout) if d]
    assert len(rendered) == sum(len(docs) for docs in manifests.values())
    images = {c["image"] for d in rendered if d["kind"] in ("Deployment", "StatefulSet") for c in d["spec"]["template"]["spec"]["containers"]}
    assert images == {"substrate:dev", "postgres:16-alpine"}


# -- the pipeline that builds and tests it ------------------------------------------------------------------------------------------------
WORKFLOWS = ROOT / ".github" / "workflows"


def test_the_workflows_pin_every_action_by_commit_and_hold_a_read_only_token_by_default():
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for use in re.findall(r"(?m)^\s*-?\s*uses:\s*(\S+)", text):
            assert re.fullmatch(r"[\w./\-]+@[0-9a-f]{40}", use), f"{path.name}: {use} is not pinned to a commit"
        workflow = load(path)
        assert workflow["permissions"] == {"contents": "read"}, path.name
        assert "pull_request_target" not in text and "docker push" not in text and "docker login" not in text, path.name


def test_ci_installs_what_the_image_installs_and_audits_every_lock_it_names():
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    workflow = load(WORKFLOWS / "ci.yml")
    assert set(workflow["jobs"]) >= {"test", "core-only", "container", "audit"}
    extras = set(re.search(r'pip install -e "\.\[([a-z0-9,]+)\]" -c requirements-dev-lock\.txt -c requirements-platform-lock\.txt', ci).group(1).split(","))
    shipped = {e for group in re.findall(r'\[([a-z0-9,]+)\]"', (DEPLOY / "Dockerfile").read_text(encoding="utf-8")) for e in group.split(",")}
    assert shipped <= extras                                                                  # the tests run against everything the image contains
    for referenced in set(re.findall(r"\b(requirements[\w\-]*\.txt)\b", ci)):
        assert (ROOT / referenced).exists(), referenced
    assert "pip-audit -r requirements-platform-lock.txt" in ci and "cyclonedx-py requirements requirements-platform-lock.txt" in ci
    assert workflow["jobs"]["container"]["steps"][1]["run"].startswith("docker build -f deploy/Dockerfile")      # built from the file the deployment uses


def test_dependabot_watches_every_kind_of_dependency_the_repository_has():
    updates = load(ROOT / ".github" / "dependabot.yml")["updates"]
    assert {(u["package-ecosystem"], u["directory"]) for u in updates} == {("pip", "/"), ("github-actions", "/"), ("docker", "/deploy"), ("docker-compose", "/deploy")}
