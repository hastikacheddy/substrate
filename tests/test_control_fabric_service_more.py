"""The ingest service, with a stand-in client so that exactly what it sends and keeps can be seen: headers, media types, redacted addresses, query checks and the default allowlist.
(Found by mutation testing; the adapters' own corners are in test_control_fabric_more.py.)"""
import json

import pytest

pytest.importorskip("sqlalchemy")

from substrate.errors import ValidationError  # noqa: E402
from substrate.control.fabric.adapters import Field, NistIonEnergetics  # noqa: E402

# -- the ingest service, with a stand-in client so that exactly what it sends and keeps can be seen -----------------------------------------------------------
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.fabric.adapters import Adapter  # noqa: E402
from substrate.control.fabric.client import Response, SafeHTTPClient  # noqa: E402
from substrate.control.fabric.service import Fabric  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402


class TabSeparated(Adapter):
    """A made-up source that answers with tab-separated text from an address with a secret in its query string."""
    id = "test.tsv"
    source = "a test"
    license = "none"
    fields = (Field("key", "str"), Field("n", "int", minimum=0))
    query_help = {}

    @property
    def hosts(self):
        return ("data.example.org",)

    def check_query(self, query):
        return {}

    def dataset_id(self, query):
        return "test.tsv"

    def url(self, query):
        return "https://data.example.org/table?api_key=SECRET"

    def parse(self, body):
        return [{"key": k, "n": int(n)} for k, n in (line.split("\t") for line in body.decode().splitlines())]


class RecordingClient:
    def __init__(self, content_type="text/tab-separated-values"):
        self.calls, self.content_type = [], content_type

    def fetch(self, url, *, headers=None):
        self.calls.append((url, headers))
        return Response(url, 200, self.content_type, b"a\t1\nb\t2\n", 1_800_000_000.0)


@pytest.fixture
def service(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    artifacts = Artifacts(registry, LocalArtifactStore(tmp_path / "s"))
    client = RecordingClient()
    return {"registry": registry, "tenant": tenant, "ada": registry.create_principal(tenant, "ada", Role.RESEARCHER), "artifacts": artifacts, "client": client,
            "fabric": Fabric(registry, artifacts, client, {"test.tsv": TabSeparated(), "nist.ion_energetics": NistIonEnergetics()})}


def test_the_headers_an_adapter_asks_for_reach_the_client(service):
    service["fabric"].ingest(service["ada"], "test.tsv", {}, headers={"X-Api-Key": "k"})
    assert service["client"].calls == [("https://data.example.org/table?api_key=SECRET", {"X-Api-Key": "k"})]


def test_a_content_type_the_artifact_store_does_not_hold_is_stored_as_plain_text_and_the_url_kept_has_no_query(service):
    dataset = service["fabric"].ingest(service["ada"], "test.tsv", {})
    meta = service["registry"].artifact_meta(service["ada"], dataset["raw_sha256"])
    assert meta["media_type"] == "text/plain" and dataset["provenance"]["content_type"] == "text/tab-separated-values"
    assert dataset["provenance"]["url"] == "https://data.example.org/table" and "SECRET" not in json.dumps(dataset["provenance"])
    audit = [e for e in service["registry"].events(f"audit:{service['tenant']}", service["tenant"]) if e["kind"] == "dataset.ingest"][0]["payload"]
    assert audit["url"] == "https://data.example.org/table" and audit["version"] == 1 and audit["checksum"] == dataset["checksum"]
    service["client"].content_type = "text/csv"
    assert service["registry"].artifact_meta(service["ada"], service["fabric"].ingest(service["ada"], "test.tsv", {})["raw_sha256"])["media_type"] == "text/plain"      # the same bytes


def test_a_query_is_checked_by_the_adapter_before_an_allowance_is_spent_or_anything_is_fetched(service):
    with pytest.raises(ValidationError, match="takes no query"):
        service["fabric"].ingest(service["ada"], "nist.ion_energetics", {"anything": 1})
    assert service["registry"].used_today(service["ada"], "dataset_ingests_per_day") == 0 and service["registry"].list_datasets() == []
    assert service["fabric"].ingest(service["ada"], "nist.ion_energetics", {})["dataset_id"] == "nist.ion_energetics"
    assert service["registry"].used_today(service["ada"], "dataset_ingests_per_day") == 1


def test_the_default_client_may_reach_the_hosts_of_the_adapters_and_no_others(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    fabric = Fabric(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "s")))
    assert fabric.client.allowed_hosts == frozenset({"pubchem.ncbi.nlm.nih.gov", "data.rcsb.org", "spec.jpl.nasa.gov"})
    assert set(fabric.adapters) == {"nist.ion_energetics", "pubchem.compound", "rcsb.entry", "jpl.spectral_lines"}
    assert isinstance(fabric.client, SafeHTTPClient)


def test_the_raw_record_of_the_bundled_set_names_its_database_and_when_it_was_retrieved(service):
    dataset = service["fabric"].ingest(service["ada"], "nist.ion_energetics", {})
    raw = json.loads(service["artifacts"].read_for(service["tenant"], dataset["raw_sha256"]))
    assert raw["source"].startswith("NIST Chemistry WebBook") and ", retrieved " in raw["source"] and len(raw["records"]) == len(dataset["records"])
    assert dataset["provenance"]["url"] == "bundled:nist.ion_energetics"
