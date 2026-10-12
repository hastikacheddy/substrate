"""Adapters and the ingest service: a malformed query or response is refused, the same data is the same version, new data is a new version, nothing is lost or rewritten."""
import http.server
import json
import threading

import pytest

pytest.importorskip("sqlalchemy")

from substrate import ValidationError  # noqa: E402
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.fabric.adapters import (Field, NistIonEnergetics, PubChemCompound, RcsbEntry, default_adapters,  # noqa: E402
                                               validate_records)
from substrate.control.fabric.client import FetchError, SafeHTTPClient  # noqa: E402
from substrate.control.fabric.service import Fabric  # noqa: E402
from substrate.control.orchestrator import Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import AuthorizationError, QuotaExceededError, Role  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import read_spec  # noqa: E402

from conftest import EXPERIMENTS  # noqa: E402

WATER = {"PropertyTable": {"Properties": [{"CID": 962, "MolecularFormula": "H2O", "MolecularWeight": "18.015", "InChIKey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N", "IUPACName": "oxidane"}]}}
ENTRY = {"rcsb_id": "4HHB", "struct": {"title": "THE CRYSTAL STRUCTURE OF HUMAN DEOXYHAEMOGLOBIN AT 1.74 A RESOLUTION"}, "exptl": [{"method": "X-RAY DIFFRACTION"}],
         "rcsb_entry_info": {"resolution_combined": [1.74], "polymer_entity_count": 2}, "rcsb_accession_info": {"initial_release_date": "1984-03-07T00:00:00+0000"}}


# -- validating records ----------------------------------------------------------------------------------------------------------------------
FIELDS = (Field("name", "str", pattern=r"[a-z]{1,5}", max_length=5), Field("n", "int", minimum=0, maximum=10), Field("x", "float", required=False, minimum=-1, maximum=1))


def test_good_records_pass_and_an_optional_field_may_be_absent():
    validate_records([{"name": "ab", "n": 3, "x": 0.5}, {"name": "c", "n": 0}], FIELDS)
    validate_records([], FIELDS)


@pytest.mark.parametrize("record, message", [
    ({"name": "ab"}, "'n' is missing"), ({"n": 1}, "'name' is missing"), ({"name": "ab", "n": 1, "extra": 1}, "unexpected field 'extra'"),
    ({"name": 5, "n": 1}, "must be text"), ({"name": "TOOLONGX", "n": 1}, "longer than 5"), ({"name": "AB", "n": 1}, "expected form"),
    ({"name": "a\nb", "n": 1}, "control characters"), ({"name": "ab", "n": "3"}, "must be a number"), ({"name": "ab", "n": True}, "must be a number"),
    ({"name": "ab", "n": 1.5}, "whole number"), ({"name": "ab", "n": -1}, "below 0"), ({"name": "ab", "n": 11}, "above 10"),
    ({"name": "ab", "n": 1, "x": float("nan")}, "finite"), ({"name": "ab", "n": 1, "x": float("inf")}, "finite"), ({"name": "ab", "n": 1, "x": 2.0}, "above 1"),
    ("not a dict", "not an object"), ({"name": "ab", "n": None}, "'n' is missing"),
])
def test_each_way_a_record_can_be_wrong_is_named(record, message):
    with pytest.raises(ValidationError, match=message):
        validate_records([record], FIELDS, owner="d")


def test_only_the_first_few_problems_are_reported_and_the_records_must_be_a_list_of_bounded_length(monkeypatch):
    with pytest.raises(ValidationError) as caught:
        validate_records([{"name": "ab"}] * 50, FIELDS)
    assert str(caught.value).count("record ") <= 5
    with pytest.raises(ValidationError, match="must be a list"):
        validate_records({"name": "ab"}, FIELDS)
    from substrate.control.fabric import adapters
    monkeypatch.setattr(adapters, "MAX_RECORDS", 3)
    with pytest.raises(ValidationError, match="4 records is more than the 3 accepted"):
        validate_records([{"name": "ab", "n": 1}] * 4, FIELDS)


# -- the adapters ----------------------------------------------------------------------------------------------------------------------------
def test_pubchem_queries_are_names_or_ids_and_nothing_that_could_alter_the_path():
    a = PubChemCompound()
    assert a.check_query({"name": "aspirin"}) == {"name": "aspirin"} and a.check_query({"cid": 2244}) == {"cid": 2244}
    assert a.check_query({"name": "2-methyl-1-propanol"}) and a.check_query({"name": "Iron(II) sulfate"}) and a.check_query({"name": "N,N-dimethylformamide"})
    for bad in ({}, {"name": "x", "cid": 1}, {"other": 1}, {"name": ""}, {"name": "../../etc/passwd"}, {"name": "a/b"}, {"name": "a;b"}, {"name": "a?b=c"}, {"name": "a#b"},
                {"name": "a%2fb"}, {"name": "a\nb"}, {"name": "x" * 101}, {"name": 5}, {"name": " leading"}, {"cid": 0}, {"cid": -3}, {"cid": "5"}, {"cid": True}, {"cid": 1.5}):
        with pytest.raises(ValidationError):
            a.check_query(bad)


def test_pubchem_urls_encode_the_name_and_use_the_documented_path():
    a = PubChemCompound()
    assert a.url({"name": "acetic acid"}) == "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/acetic%20acid/property/MolecularFormula,MolecularWeight,InChIKey,IUPACName/JSON"
    assert a.url({"cid": 962}).startswith("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/962/property/") and a.hosts == ("pubchem.ncbi.nlm.nih.gov",)
    assert a.dataset_id({"name": "Acetic Acid"}) == "pubchem.compound.acetic-acid" and a.dataset_id({"cid": 962}) == "pubchem.compound.cid-962"


def test_pubchem_responses_are_read_into_typed_records():
    a = PubChemCompound()
    records = a.parse(json.dumps(WATER).encode())
    assert records == [{"cid": 962, "formula": "H2O", "molecular_weight": 18.015, "inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N", "iupac_name": "oxidane"}]
    validate_records(records, a.fields)
    without_name = json.loads(json.dumps(WATER))
    del without_name["PropertyTable"]["Properties"][0]["IUPACName"]
    assert "iupac_name" not in a.parse(json.dumps(without_name).encode())[0]


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"PropertyTable": 5}', b'{"PropertyTable": {"Properties": "x"}}', b'{"PropertyTable": {"Properties": [{"CID": 1}]}}',
                                  b'{"PropertyTable": {"Properties": [{"CID": 1, "MolecularFormula": "H", "MolecularWeight": "heavy", "InChIKey": "K"}]}}', b'{"x": NaN}', b"\xff\xfe"])
def test_responses_that_are_not_the_documented_shape_are_refused(body):
    with pytest.raises(ValidationError):
        PubChemCompound().parse(body)


def test_a_response_with_values_outside_the_declared_ranges_fails_validation_even_if_it_parses():
    bad = json.loads(json.dumps(WATER))
    bad["PropertyTable"]["Properties"][0]["MolecularWeight"] = "-5"
    records = PubChemCompound().parse(json.dumps(bad).encode())
    with pytest.raises(ValidationError, match="below 0"):
        validate_records(records, PubChemCompound.fields)
    bad["PropertyTable"]["Properties"][0].update({"MolecularWeight": "18", "InChIKey": "not-a-key"})
    with pytest.raises(ValidationError, match="expected form"):
        validate_records(PubChemCompound().parse(json.dumps(bad).encode()), PubChemCompound.fields)


def test_rcsb_entries_are_four_characters_and_normalised_to_upper_case():
    a = RcsbEntry()
    assert a.check_query({"entry": "4hhb"}) == {"entry": "4HHB"} and a.url({"entry": "1abc"}) == "https://data.rcsb.org/rest/v1/core/entry/1ABC"
    assert a.dataset_id({"entry": "4HHB"}) == "rcsb.entry.4hhb" and a.hosts == ("data.rcsb.org",)
    for bad in ({}, {"entry": "4HH"}, {"entry": "4HHB1"}, {"entry": "AHHB"}, {"entry": "../x"}, {"entry": "4H/B"}, {"entry": 4}, {"entry": "4HHB", "x": 1}):
        with pytest.raises(ValidationError):
            a.check_query(bad)


def test_rcsb_responses_become_one_record_and_optional_parts_may_be_missing():
    a = RcsbEntry()
    record = a.parse(json.dumps(ENTRY).encode())
    assert record == [{"id": "4HHB", "title": ENTRY["struct"]["title"], "method": "X-RAY DIFFRACTION", "resolution_angstrom": 1.74, "release_date": "1984-03-07", "polymer_entity_count": 2}]
    validate_records(record, a.fields)
    minimal = a.parse(json.dumps({"rcsb_id": "1ABC", "struct": {"title": "t"}}).encode())
    assert minimal == [{"id": "1ABC", "title": "t"}]
    for bad in (b"{}", b'{"rcsb_id": "1ABC"}', b"[]", b"junk"):
        with pytest.raises(ValidationError):
            a.parse(bad)


def test_the_bundled_nist_set_is_an_adapter_with_no_network_and_valid_records():
    a = NistIonEnergetics()
    assert a.network is False and a.hosts == () and a.check_query({}) == {} and a.dataset_id({}) == "nist.ion_energetics"
    records = a.parse(a.read())
    validate_records(records, a.fields)
    assert len(records) == 10 and {r["id"] for r in records} >= {"pa_h2o", "pa_nh3", "acidity_hf"} and all(r["unit"] == "kJ/mol" for r in records)
    with pytest.raises(ValidationError, match="no query"):
        a.check_query({"x": 1})


def test_the_adapters_describe_themselves_for_the_api():
    info = {d["id"]: d for d in (a.describe() for a in default_adapters().values())}
    assert set(info) == {"nist.ion_energetics", "pubchem.compound", "rcsb.entry", "jpl.spectral_lines"} and info["nist.ion_energetics"]["network"] is False
    assert info["pubchem.compound"]["hosts"] == ["pubchem.ncbi.nlm.nih.gov"] and info["rcsb.entry"]["license"].startswith("PDB archive data")


# -- the service, against a local server ---------------------------------------------------------------------------------------------------
class Remote(http.server.BaseHTTPRequestHandler):
    routes: dict = {}
    hits: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        Remote.hits.append(self.path)
        status, body = Remote.routes.get(self.path.split("?")[0], (404, b"{}"))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def remote():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Remote)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def world(tmp_path, remote):
    Remote.routes = {"/rest/pug/compound/name/water/property/MolecularFormula,MolecularWeight,InChIKey,IUPACName/JSON": (200, json.dumps(WATER).encode()),
                     "/rest/v1/core/entry/4HHB": (200, json.dumps(ENTRY).encode())}
    Remote.hits = []
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    artifacts = Artifacts(registry, LocalArtifactStore(tmp_path / "s"))
    adapters = {a.id: a for a in (NistIonEnergetics(), PubChemCompound(remote), RcsbEntry(remote))}
    fabric = Fabric(registry, artifacts, SafeHTTPClient({"127.0.0.1"}, allow_loopback=True, sleep=lambda s: None), adapters)
    return {"registry": registry, "tenant": tenant, "admin": admin, "artifacts": artifacts, "fabric": fabric, "tmp": tmp_path,
            "ada": registry.create_principal(tenant, "ada", Role.RESEARCHER), "vera": registry.create_principal(tenant, "vera", Role.VIEWER),
            "olga": registry.create_principal(tenant, "olga", Role.OPERATOR)}


def test_an_ingest_stores_the_records_the_raw_bytes_and_where_they_came_from(world):
    dataset = world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "water"})
    assert (dataset["dataset_id"], dataset["version"]) == ("pubchem.compound.water", 1) and dataset["records"][0]["formula"] == "H2O"
    assert dataset["source"].startswith("PubChem") and dataset["license"].startswith("PubChem data are open") and dataset["schema"]["adapter"] == "pubchem.compound"
    provenance = dataset["provenance"]
    assert provenance["query"] == {"name": "water"} and provenance["ingested_by"] == world["ada"].id and provenance["adapter_version"] == "1"
    assert provenance["url"].startswith("http://127.0.0.1:") and "?" not in provenance["url"] and provenance["raw_bytes"] == len(json.dumps(WATER).encode())
    assert world["artifacts"].read_for(world["tenant"], dataset["raw_sha256"]) == json.dumps(WATER).encode()
    event = [e for e in world["registry"].events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "dataset.ingest"][0]
    assert event["payload"]["dataset"] == "pubchem.compound.water" and event["payload"]["checksum"] == dataset["checksum"]


def test_ingesting_the_same_data_again_is_the_same_version_and_new_data_is_a_new_one(world):
    first = world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "water"})
    again = world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "water"})
    assert again["version"] == first["version"] == 1 and len(world["registry"].list_datasets()) == 1
    changed = json.loads(json.dumps(WATER))
    changed["PropertyTable"]["Properties"][0]["MolecularWeight"] = "18.0153"
    Remote.routes["/rest/pug/compound/name/water/property/MolecularFormula,MolecularWeight,InChIKey,IUPACName/JSON"] = (200, json.dumps(changed).encode())
    newer = world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "water"})
    assert newer["version"] == 2 and newer["records"][0]["molecular_weight"] == 18.0153
    assert world["registry"].get_dataset("pubchem.compound.water", 1)["records"][0]["molecular_weight"] == 18.015           # the old version is exactly as it was
    assert newer["checksum"] != first["checksum"] and newer["raw_sha256"] != first["raw_sha256"]


def test_the_bundled_set_is_ingested_without_any_request(world):
    dataset = world["fabric"].ingest(world["ada"], "nist.ion_energetics")
    assert dataset["dataset_id"] == "nist.ion_energetics" and len(dataset["records"]) == 10 and dataset["provenance"]["url"] == "bundled:nist.ion_energetics" and Remote.hits == []


def test_rcsb_entries_are_ingested_the_same_way(world):
    dataset = world["fabric"].ingest(world["ada"], "rcsb.entry", {"entry": "4hhb"})
    assert dataset["dataset_id"] == "rcsb.entry.4hhb" and dataset["records"][0]["resolution_angstrom"] == 1.74 and Remote.hits == ["/rest/v1/core/entry/4HHB"]


def test_only_roles_that_may_ingest_can_and_a_refused_request_costs_nothing(world):
    for who in ("vera", "olga"):
        with pytest.raises(AuthorizationError):
            world["fabric"].ingest(world[who], "pubchem.compound", {"name": "water"})
    for adapter, query in (("pubchem.compound", {"name": "../../etc/passwd"}), ("pubchem.compound", {}), ("nope", {}), ("rcsb.entry", {"entry": "x"})):
        with pytest.raises(ValidationError):
            world["fabric"].ingest(world["ada"], adapter, query)
    assert Remote.hits == [] and world["registry"].used_today(world["ada"], "dataset_ingests_per_day") == 0 and world["registry"].list_datasets() == []


def test_the_daily_allowance_stops_the_fetch_before_it_is_made(world):
    world["registry"].set_quota(world["admin"], "dataset_ingests_per_day", 1)
    world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "water"})
    hits = len(Remote.hits)
    with pytest.raises(QuotaExceededError, match="dataset_ingests_per_day"):
        world["fabric"].ingest(world["ada"], "rcsb.entry", {"entry": "4HHB"})
    assert len(Remote.hits) == hits


def test_a_response_that_fails_validation_stores_nothing_not_even_the_raw_bytes(world):
    bad = json.loads(json.dumps(WATER))
    bad["PropertyTable"]["Properties"][0]["InChIKey"] = "nonsense"
    Remote.routes["/rest/pug/compound/name/water/property/MolecularFormula,MolecularWeight,InChIKey,IUPACName/JSON"] = (200, json.dumps(bad).encode())
    with pytest.raises(ValidationError, match="expected form"):
        world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "water"})
    assert world["registry"].list_datasets() == [] and world["registry"].artifact_bytes_total(world["tenant"]) == 0


def test_an_unavailable_source_is_a_fetch_error_and_stores_nothing(world):
    with pytest.raises(FetchError, match="answered 404"):
        world["fabric"].ingest(world["ada"], "pubchem.compound", {"name": "nothing-here"})
    assert world["registry"].list_datasets() == []


def test_a_fabric_pointed_at_a_host_that_is_not_on_its_allowlist_cannot_reach_it(world):
    fabric = Fabric(world["registry"], world["artifacts"], SafeHTTPClient({"data.example.org"}), {"pubchem.compound": PubChemCompound("https://pubchem.ncbi.nlm.nih.gov")})
    with pytest.raises(FetchError, match="not on the allowlist"):
        fabric.ingest(world["ada"], "pubchem.compound", {"name": "water"})


def test_an_experiment_pinned_to_an_ingested_dataset_runs_and_reports_its_source_and_licence(world):
    dataset = world["fabric"].ingest(world["ada"], "nist.ion_energetics")
    registry = world["registry"]
    registry.set_quota(world["admin"], "concurrent_runs", 5)
    exp = registry.add_experiment(world["ada"], "pinned", read_spec(EXPERIMENTS / "proton_transfer.yaml"), [{"id": dataset["dataset_id"], "version": dataset["version"]}])
    orchestrator = Orchestrator(registry, world["artifacts"])
    run = orchestrator.submit(world["ada"], exp["id"], n_samples=0)
    Worker(orchestrator).run_until_idle()
    assert orchestrator.describe(world["ada"], run["id"])["state"] == "succeeded"
    report = [a for a in registry.run_artifacts(world["admin"], run["id"]) if a["name"] == "report.json"][0]
    pinned = world["artifacts"].read_json(world["tenant"], report["sha256"])["datasets"][0]
    assert pinned["id"] == "nist.ion_energetics" and pinned["checksum"] == dataset["checksum"] and pinned["license"].startswith("NIST") and "WebBook" in pinned["source"]
