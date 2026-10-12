"""The data adapters against the real services: PubChem, RCSB and NASA JPL.

Skipped unless SUBSTRATE_TEST_LIVE=1: a test suite should not depend on three outside servers being up, on the network, or on their rate limits (the client waits
between requests to the same host). Run it when an adapter is touched, or now and then to see whether a source has changed its format; the adapters' field checks are
what would notice, and a failure here names the field. The things asserted are facts that do not move: the formula of aspirin, the structure of haemoglobin, and the two
best-known water lines (the 22 GHz and 183 GHz masers), not whatever a database returned on the day this was written.
"""
import os
import re

import pytest

pytest.importorskip("sqlalchemy")

from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.fabric.adapters import default_adapters  # noqa: E402
from substrate.control.fabric.client import SafeHTTPClient  # noqa: E402
from substrate.control.fabric.service import Fabric  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("SUBSTRATE_TEST_LIVE") != "1", reason="live fetches are off: set SUBSTRATE_TEST_LIVE=1")


@pytest.fixture
def lab(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    adapters = default_adapters()
    client = SafeHTTPClient({h for a in adapters.values() for h in a.hosts if h}, min_interval_s=0.5)        # the real client: HTTPS only, public addresses only
    fabric = Fabric(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "store")), client, adapters)
    return {"fabric": fabric, "registry": registry, "ada": registry.create_principal(tenant, "ada", Role.RESEARCHER)}


def test_pubchem_aspirin_by_compound_id(lab):
    dataset = lab["fabric"].ingest(lab["ada"], "pubchem.compound", {"cid": 2244})
    record = lab["registry"].get_dataset(dataset["dataset_id"])["records"][0]
    assert (record["cid"], record["formula"]) == (2244, "C9H8O4") and abs(record["molecular_weight"] - 180.16) < 0.05
    assert record["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_pubchem_water_by_name(lab):
    dataset = lab["fabric"].ingest(lab["ada"], "pubchem.compound", {"name": "water"})
    record = lab["registry"].get_dataset(dataset["dataset_id"])["records"][0]
    assert record["cid"] == 962 and record["formula"] == "H2O" and abs(record["molecular_weight"] - 18.015) < 0.01


def test_rcsb_haemoglobin_4hhb(lab):
    dataset = lab["fabric"].ingest(lab["ada"], "rcsb.entry", {"entry": "4hhb"})
    record = lab["registry"].get_dataset(dataset["dataset_id"])["records"][0]
    assert record["id"] == "4HHB" and re.search(r"HA?EMOGLOBIN", record["title"].upper()) and record["method"] == "X-RAY DIFFRACTION"     # the entry says "deoxyhaemoglobin"
    assert 1.5 < record["resolution_angstrom"] < 2.0


def test_jpl_water_has_the_two_best_known_lines_and_they_are_flagged_measured(lab):
    dataset = lab["fabric"].ingest(lab["ada"], "jpl.spectral_lines", {"tag": 18003})
    stored = lab["registry"].get_dataset(dataset["dataset_id"])
    assert stored["provenance"]["content_type"] == "text/plain" and len(stored["records"]) > 1000
    lines = {round(r["frequency_mhz"], 3): r for r in stored["records"]}
    maser_22, line_183 = lines[22235.080], lines[183310.087]
    assert maser_22["measured"] == 1 and maser_22["upper_qn"].split() == ["6", "1", "6", "0"] and maser_22["lower_qn"].split() == ["5", "2", "3", "0"]
    assert line_183["measured"] == 1 and line_183["upper_qn"].split() == ["3", "1", "3", "0"] and line_183["lower_qn"].split() == ["2", "2", "0", "0"]
    assert all(r["tag"] == 18003 for r in stored["records"])


def test_jpl_carbon_monoxide_ground_state_line_is_at_115_gigahertz(lab):
    dataset = lab["fabric"].ingest(lab["ada"], "jpl.spectral_lines", {"tag": 28001})
    first = min(lab["registry"].get_dataset(dataset["dataset_id"])["records"], key=lambda r: r["frequency_mhz"])
    assert abs(first["frequency_mhz"] - 115271.2018) < 1e-3 and first["lower_energy_cm1"] == 0.0 and first["measured"] == 1
