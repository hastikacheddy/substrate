"""The S3 artifact store against a real S3-compatible server.

Skipped unless SUBSTRATE_TEST_S3_ENDPOINT names one (a throwaway server, for example SeaweedFS: `weed server -s3 -s3.config=identities.json`) and the usual
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY name an identity that may create buckets on it. The fake client in test_control_artifacts.py checks the logic; this checks that the
logic matches what a real server and boto3 do (which is how the 403-as-"not found" mistake was found).
"""
import os
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
boto3 = pytest.importorskip("boto3")

from botocore.config import Config  # noqa: E402

from substrate.control.artifacts import Artifacts, ArtifactError, S3ArtifactStore  # noqa: E402
from substrate.control.orchestrator import Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import read_spec  # noqa: E402

ENDPOINT = os.environ.get("SUBSTRATE_TEST_S3_ENDPOINT")
pytestmark = pytest.mark.skipif(not ENDPOINT, reason="SUBSTRATE_TEST_S3_ENDPOINT is not set (a throwaway S3-compatible server, with AWS_* credentials for it)")
FAST = Config(retries={"max_attempts": 1}, connect_timeout=3, read_timeout=10)
EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"


def make_client(**overrides):
    options = {"endpoint_url": ENDPOINT, "region_name": "us-east-1", "config": FAST, **overrides}
    return boto3.client("s3", **options)


@pytest.fixture
def bucket():
    name = "t-" + uuid.uuid4().hex[:16]
    client = make_client()
    client.create_bucket(Bucket=name)
    yield name
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=name):
        for item in page.get("Contents", []):
            client.delete_object(Bucket=name, Key=item["Key"])
    client.delete_bucket(Bucket=name)


@pytest.fixture
def store(bucket):
    return S3ArtifactStore.from_boto3(bucket, "artifacts", endpoint_url=ENDPOINT, region_name="us-east-1", config=FAST)


def test_put_get_exists_round_trip_and_an_existing_object_is_not_written_again(store, bucket):
    sha = store.put("T-0001", b"payload")
    assert store.exists("T-0001", sha) and store.get("T-0001", sha) == b"payload"
    key = store._key("T-0001", sha)
    first = make_client().head_object(Bucket=bucket, Key=key)
    time.sleep(1.1)
    assert store.put("T-0001", b"payload") == sha
    assert make_client().head_object(Bucket=bucket, Key=key)["LastModified"] == first["LastModified"]


def test_a_missing_object_is_missing_and_each_tenant_has_its_own_keys(store):
    sha = store.put("T-0001", b"payload")
    assert store.exists("T-0002", sha) is False and store.exists("T-0001", "c" * 64) is False
    with pytest.raises(ArtifactError, match="no artifact"):
        store.get("T-0002", sha)
    with pytest.raises(ArtifactError, match="no artifact"):
        store.get("T-0001", "c" * 64)


def test_bytes_changed_behind_the_stores_back_are_caught_on_read(store, bucket):
    sha = store.put("T-0001", b"payload")
    make_client().put_object(Bucket=bucket, Key=store._key("T-0001", sha), Body=b"changed")
    with pytest.raises(ArtifactError, match="checksum"):
        store.get("T-0001", sha)


def test_a_few_megabytes_and_binary_content_survive(store):
    data = os.urandom(5 * 1024 * 1024)
    sha = store.put("T-0001", data)
    assert store.get("T-0001", sha) == data


def test_wrong_credentials_are_an_error_and_not_a_missing_artifact(bucket):
    good = S3ArtifactStore.from_boto3(bucket, endpoint_url=ENDPOINT, region_name="us-east-1", config=FAST)
    sha = good.put("T-0001", b"payload")
    bad = S3ArtifactStore.from_boto3(bucket, endpoint_url=ENDPOINT, region_name="us-east-1", config=FAST, aws_access_key_id="AKINVALID0000", aws_secret_access_key="wrong" * 8)
    from botocore.exceptions import ClientError
    with pytest.raises(ClientError):
        bad.exists("T-0001", sha)
    with pytest.raises(ClientError):
        bad.get("T-0001", sha)
    with pytest.raises(ClientError):
        bad.put("T-0001", b"other")


def test_an_unreachable_server_is_an_error_and_not_a_missing_artifact(bucket):
    from botocore.exceptions import EndpointConnectionError
    down = S3ArtifactStore.from_boto3(bucket, endpoint_url="http://127.0.0.1:9", region_name="us-east-1", config=FAST, aws_access_key_id="x", aws_secret_access_key="y")
    start = time.perf_counter()
    with pytest.raises(EndpointConnectionError):
        down.exists("T-0001", "a" * 64)
    with pytest.raises(EndpointConnectionError):
        down.get("T-0001", "a" * 64)
    assert time.perf_counter() - start < 30


def test_the_cli_builds_the_same_store_from_an_s3_location(bucket, monkeypatch):
    from substrate.control.cli import artifact_store
    monkeypatch.setenv("SUBSTRATE_S3_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    cli_store = artifact_store(f"s3://{bucket}/from-cli")
    sha = cli_store.put("T-0001", b"via the cli's factory")
    assert make_client().get_object(Bucket=bucket, Key=f"from-cli/T-0001/{sha[:2]}/{sha}")["Body"].read() == b"via the cli's factory"


def test_a_whole_run_keeps_its_artifacts_in_the_bucket_and_reproduces(bucket, tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 50)
    ada = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    store = S3ArtifactStore.from_boto3(bucket, "runs", endpoint_url=ENDPOINT, region_name="us-east-1", config=FAST)
    orchestrator = Orchestrator(registry, Artifacts(registry, store))
    experiment = registry.add_experiment(ada, "proton_transfer", read_spec(EXPERIMENTS / "proton_transfer.yaml"))["id"]
    run = orchestrator.submit(ada, experiment, n_samples=0)["id"]
    assert Worker(orchestrator, queues=("cpu",)).run_until_idle() == 7
    description = orchestrator.describe(ada, run)
    assert description["state"] == "succeeded"
    listed = registry.run_artifacts(ada, run)
    assert {a["name"] for a in listed} >= {"report.json", "report.md", "final_observables.json", "status.json"}
    objects = [o["Key"] for page in make_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="runs/") for o in page.get("Contents", [])]
    assert len(objects) == len({a["sha256"] for a in listed}) and all(k.startswith(f"runs/{tenant}/") for k in objects)
    again = orchestrator.reproduce(ada, run)["id"]
    Worker(orchestrator, queues=("cpu",)).run_until_idle()
    verdict = [r for r in registry.reproductions_of(ada, run)][0]
    assert verdict["state"] == "succeeded" and verdict["status"]["reproducibility"]["state"] == "PASS" and verdict["id"] == again
