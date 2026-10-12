"""The JPL spectral-line adapter, against lines taken from the real catalog files (c018003.cat, water, and c028001.cat, carbon monoxide; NASA JPL,
https://spec.jpl.nasa.gov) and against the ways a file can be wrong. The expected values are the catalog's own: the 22.235 GHz and 183.310 GHz water lines are the
well-known ones, with the quantum-number assignments of the textbooks (6(1,6)-5(2,3) and 3(1,3)-2(2,0))."""
import pytest

pytest.importorskip("sqlalchemy")

from substrate.errors import ValidationError  # noqa: E402
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.fabric.adapters import JplSpectralLines, default_adapters, validate_records  # noqa: E402
from substrate.control.fabric.client import Response  # noqa: E402
from substrate.control.fabric.service import Fabric  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402

WATER = [
    '    8006.5805  2.8510-18.6204 3 6219.6192 45  18003140422 418 0    21 715 0',
    '   22235.0798  0.0001 -5.8825 3  446.5107 39 -180031404 6 1 6 0     5 2 3 0',
    '  183310.0870  0.0010 -3.6463 3  136.1639  7 -180031404 3 1 3 0     2 2 0 0',
]
CO = [
    '  115271.2018   .0005 -5.0105 2     .0000  3 -28001 101 1           0',
    '  230538.0000   .0005 -4.1197 2    3.8450  5 -28001 101 2           1',
]


def body(lines):
    return ("\n".join(line.ljust(80) for line in lines) + "\n").encode("ascii")


adapter = JplSpectralLines()


# -- reading the real lines ---------------------------------------------------------------------------------------------------------------------
def test_the_22_ghz_water_line_is_read_field_by_field():
    record = adapter.parse(body(WATER))[1]
    assert record == {"frequency_mhz": 22235.0798, "uncertainty_mhz": 0.0001, "log10_intensity_nm2_mhz_300k": -5.8825, "rotational_dof": 3, "lower_energy_cm1": 446.5107,
                      "upper_degeneracy": 39, "tag": 18003, "measured": 1, "qn_format": 1404, "upper_qn": " 6 1 6 0", "lower_qn": " 5 2 3 0"}


def test_the_sign_of_the_tag_says_measured_or_predicted_and_the_tag_itself_is_always_positive():
    records = adapter.parse(body(WATER))
    assert [(r["tag"], r["measured"]) for r in records] == [(18003, 0), (18003, 1), (18003, 1)]
    assert [r["frequency_mhz"] for r in records] == [8006.5805, 22235.0798, 183310.087]


def test_a_value_written_without_the_zero_before_the_point_is_read_as_the_fortran_format_writes_it():
    records = adapter.parse(body(CO))
    assert records[0]["uncertainty_mhz"] == 0.0005 and records[0]["lower_energy_cm1"] == 0.0 and records[1]["lower_energy_cm1"] == 3.845
    assert (records[0]["rotational_dof"], records[0]["qn_format"], records[0]["upper_qn"], records[0]["lower_qn"]) == (2, 101, " 1", " 0")


def test_the_records_pass_the_declared_fields_and_the_query_check_and_a_trailing_blank_line_is_ignored():
    records = adapter.parse(body(WATER + CO[:0]) + b"\n  \n")
    validate_records(records, adapter.fields, owner="jpl")
    adapter.verify(records, {"tag": 18003})
    assert len(records) == 3


def test_the_url_the_id_and_the_host_come_from_the_tag_only():
    assert adapter.url({"tag": 18003}) == "https://spec.jpl.nasa.gov/ftp/pub/catalog/c018003.cat"
    assert adapter.url({"tag": 1001}) == "https://spec.jpl.nasa.gov/ftp/pub/catalog/c001001.cat"
    assert adapter.dataset_id({"tag": 28001}) == "jpl.spectral_lines.028001" and adapter.hosts == ("spec.jpl.nasa.gov",)
    assert adapter.assume_type == "text/plain" and adapter.network is True
    assert JplSpectralLines("http://127.0.0.1:9/").url({"tag": 5}) == "http://127.0.0.1:9/ftp/pub/catalog/c000005.cat"


def test_the_adapter_is_registered_and_describes_itself():
    assert "jpl.spectral_lines" in default_adapters()
    description = default_adapters()["jpl.spectral_lines"].describe()
    assert description["hosts"] == ["spec.jpl.nasa.gov"] and "tag" in description["query"] and "NASA" in description["source"]


# -- what is refused ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("query", [{}, {"tag": 0}, {"tag": 1_000_000}, {"tag": -5}, {"tag": "18003"}, {"tag": 1.5}, {"tag": True}, {"tag": 18003, "x": 1}, {"species": 18003},
                                   {"tag": None}])
def test_a_query_that_is_not_a_species_tag_is_refused(query):
    with pytest.raises(ValidationError, match="tag"):
        adapter.check_query(query)


def test_a_file_of_another_species_is_refused_by_name():
    records = adapter.parse(body(WATER[:1] + CO[:1]))
    with pytest.raises(ValidationError, match="species 28001, not the species 18003"):
        adapter.verify(records, {"tag": 18003})


@pytest.mark.parametrize("text, message", [
    (b"", "no lines"), (b"\n \n", "no lines"),
    ("   22235.0798  0.0001 -5.8825 3  446.5107 39 -180031404 6 1 6 0     5 2 3 0 é".encode("utf-8"), "ASCII"),
    (b"short line 1 2 3\n", "not a catalog line"), (("x" * 120).encode(), "not a catalog line"),
    (body([WATER[1].replace("22235.0798", "22235x0798")]), "fixed-width layout"),                       # a letter where a number belongs
    (body([WATER[1].replace(" 39 ", " 3.9")]), "fixed-width layout"),                                     # a decimal point in a whole-number field
    (body([WATER[1].replace("-5.8825", "-5.88e5")]), "fixed-width layout"),
    (body([WATER[1].replace("22235.0798", "         nan")]), "fixed-width layout"),
    (body([WATER[1].replace("22235.0798", "1_235.0798 ")]), "fixed-width layout"),                       # Python would read this one; the catalog never writes it
    (body([WATER[1].replace("0.0001", "      ")]), "fixed-width layout"),
])
def test_a_file_that_does_not_have_the_catalogs_layout_is_refused_with_its_line_number(text, message):
    with pytest.raises(ValidationError, match=message):
        adapter.parse(text)


def test_the_error_names_the_line_that_is_wrong():
    with pytest.raises(ValidationError, match="line 3 "):
        adapter.parse(body(WATER[:2] + ["not a line at all, but long enough to pass the length check, 1234567890123456789012345678"]))


def test_a_physically_impossible_value_is_refused_by_the_field_checks():
    for replace, field in (("  0.0001", "uncertainty_mhz"), (" -5.8825", "log10_intensity_nm2_mhz_300k"), (" 3  446.5107", "lower_energy_cm1")):
        bad = {"  0.0001": " -0.0001", " -5.8825": " 99.0000", " 3  446.5107": " 3 -446.5107"}[replace]
        records = adapter.parse(body([WATER[1].replace(replace, bad)]))
        with pytest.raises(ValidationError, match=field):
            validate_records(records, adapter.fields, owner="jpl")


def test_a_file_with_more_lines_than_the_limit_is_refused_before_it_is_parsed():
    from substrate.control.fabric import adapters
    big = ("\n".join([WATER[1]] * (adapters.MAX_RECORDS + 1))).encode()
    with pytest.raises(ValidationError, match="more than"):
        adapter.parse(big)


# -- through the service, with the stand-in client the other service tests use ---------------------------------------------------------------------
class NoTypeClient:
    """Answers the way JPL's server does: with the file and no content type of its own, so the service must be the one to say what to assume."""

    def __init__(self, text):
        self.text, self.calls = text, []

    def fetch(self, url, *, headers=None, assume_type=None):
        self.calls.append((url, assume_type))
        return Response(url, 200, assume_type or "", self.text, 1_800_000_000.0)


@pytest.fixture
def service(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    return {"registry": registry, "artifacts": Artifacts(registry, LocalArtifactStore(tmp_path / "s")), "ada": registry.create_principal(tenant, "ada", Role.RESEARCHER)}


def ingest(service, text, query=None):
    client = NoTypeClient(text)
    fabric = Fabric(service["registry"], service["artifacts"], client, {"jpl.spectral_lines": adapter})
    return fabric.ingest(service["ada"], "jpl.spectral_lines", query or {"tag": 18003}), client


def test_ingesting_a_catalog_file_keeps_the_records_the_raw_file_and_where_it_came_from(service):
    dataset, client = ingest(service, body(WATER))
    assert client.calls == [("https://spec.jpl.nasa.gov/ftp/pub/catalog/c018003.cat", "text/plain")]
    assert dataset["dataset_id"] == "jpl.spectral_lines.018003" and dataset["version"] == 1
    stored = service["registry"].get_dataset("jpl.spectral_lines.018003")
    assert len(stored["records"]) == 3 and stored["records"][2]["frequency_mhz"] == 183310.087
    assert stored["provenance"]["content_type"] == "text/plain" and stored["provenance"]["raw_bytes"] == len(body(WATER))
    assert stored["provenance"]["url"] == "https://spec.jpl.nasa.gov/ftp/pub/catalog/c018003.cat" and "NASA JPL" in stored["source"]
    assert service["artifacts"].read(service["ada"], stored["raw_sha256"])[0] == body(WATER)


def test_a_file_of_the_wrong_species_is_not_kept(service):
    with pytest.raises(ValidationError, match="species 28001"):
        ingest(service, body(CO), {"tag": 18003})
    assert service["registry"].get_dataset("jpl.spectral_lines.018003") is None


def test_a_file_that_fails_the_checks_is_not_kept_either(service):
    with pytest.raises(ValidationError, match="fixed-width"):
        ingest(service, body([WATER[1].replace("22235.0798", "22235x0798")]))
    assert service["registry"].list_datasets() == []
