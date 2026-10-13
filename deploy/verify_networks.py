"""Check that every running container of the compose project is attached to exactly the networks docker-compose.yml gives its service.

    python deploy/verify_networks.py                  # exit 0 if all are right, 1 if any container is missing a network (or has an extra one)

Why this exists: Docker Engine 25.0 with Compose 2.24 sometimes creates a container attached to only some of its networks, without an error (a two-service alpine project
shows it about two times in ten). The container then cannot reach what it was meant to, or be reached on its published port. Newer Docker does not do this. The fix for a
container named here is `docker compose -f deploy/docker-compose.yml up -d --force-recreate <service>`.

Standard library only; it needs the docker CLI. It reads and changes nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

COMPOSE = Path(__file__).resolve().parent / "docker-compose.yml"


def expected_networks(config: dict, project: str) -> dict[str, set[str]]:
    """{service: {project network names}} from `docker compose config --format json`."""
    wanted = {}
    for name, service in config["services"].items():
        networks = service.get("networks") or {"default": None}
        wanted[name] = {(config["networks"].get(n) or {}).get("name") or f"{project}_{n}" for n in networks}
    return wanted


def mismatches(wanted: dict[str, set[str]], containers: list[dict]) -> list[str]:
    """One sentence per container whose networks differ from its service's. `containers` are `docker inspect` records."""
    problems = []
    for record in containers:
        service = record["Config"]["Labels"].get("com.docker.compose.service")
        actual = set(record["NetworkSettings"]["Networks"])
        if service not in wanted:
            continue
        missing, extra = wanted[service] - actual, actual - wanted[service]
        if missing or extra:
            parts = ([f"missing {', '.join(sorted(missing))}"] if missing else []) + ([f"has unexpected {', '.join(sorted(extra))}"] if extra else [])
            problems.append(f"{record['Name'].lstrip('/')} ({service}): " + "; ".join(parts))
    return problems


def docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise SystemExit(f"docker {' '.join(args[:2])} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def main() -> int:
    config = json.loads(docker("compose", "-f", str(COMPOSE), "config", "--format", "json"))
    ids = docker("compose", "-f", str(COMPOSE), "ps", "-q").split()
    if not ids:
        print("no containers of this project are running")
        return 1
    records = json.loads(docker("inspect", *ids))
    problems = mismatches(expected_networks(config, config["name"]), records)
    for problem in problems:
        print(problem)
    if not problems:
        print(f"all {len(records)} containers are attached to the networks the compose file gives them")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
