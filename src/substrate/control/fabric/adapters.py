"""Adapters: how one outside dataset is asked for, read, and checked.

An adapter knows four things about its source: how to build the request URL from a query (and to refuse a query that is not what the source accepts), which hosts it
needs, how to turn the response into flat records, and what every record must look like. Everything that comes back is untrusted: the records are checked against the
adapter's declared fields (type, range, pattern, length, finiteness) and an unexpected field or a malformed value is an error, not something quietly passed on.

The adapters were written from the sources' published documentation. The unit tests run them against responses of the documented shape (for the JPL catalog, lines taken
from the real file); PubChem, RCSB and JPL were also run against the live services once, and `tests/test_control_fabric_live.py` repeats that when SUBSTRATE_TEST_LIVE=1
(it is off by default, because a test suite should not depend on three outside servers). A source can change its format without notice: the field checks are what would
notice it.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from ...errors import ValidationError

MAX_RECORDS = 100_000
MAX_TEXT = 2000


@dataclass(frozen=True)
class Field:
    name: str
    type: str                                   # "str", "float" or "int"
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None
    max_length: int = MAX_TEXT

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def validate_records(records, fields: tuple[Field, ...], *, owner: str = "dataset") -> None:
    """Raise ValidationError, naming the first few problems, unless every record has exactly the declared fields with acceptable values."""
    if not isinstance(records, list):
        raise ValidationError(f"{owner}: the records must be a list")
    if len(records) > MAX_RECORDS:
        raise ValidationError(f"{owner}: {len(records):,} records is more than the {MAX_RECORDS:,} accepted")
    known = {f.name: f for f in fields}
    problems = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(f"record {index} is not an object")
            continue
        for name in sorted(set(record) - set(known)):
            problems.append(f"record {index}: unexpected field '{name}'")
        for field in fields:
            value = record.get(field.name)
            if value is None:
                if field.required:
                    problems.append(f"record {index}: '{field.name}' is missing")
                continue
            problem = _check(field, value)
            if problem:
                problems.append(f"record {index}: '{field.name}' {problem}")
        if len(problems) >= 5:
            break
    if problems:
        raise ValidationError(f"{owner}: " + "; ".join(problems[:5]))


def _check(field: Field, value) -> str | None:
    if field.type == "str":
        if not isinstance(value, str):
            return "must be text"
        if len(value) > field.max_length:
            return f"is longer than {field.max_length} characters"
        if any(ord(c) < 32 for c in value):
            return "contains control characters"
        if field.pattern and not re.fullmatch(field.pattern, value):
            return "does not have the expected form"
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be a number"
    if field.type == "int" and (isinstance(value, float) and value != math.floor(value)):
        return "must be a whole number"
    if not math.isfinite(value):
        return "must be finite"
    if field.minimum is not None and value < field.minimum:
        return f"is below {field.minimum:g}"
    if field.maximum is not None and value > field.maximum:
        return f"is above {field.maximum:g}"
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "x"


class Adapter:
    id: str = ""
    source: str = ""
    license: str = ""
    version: str = "1"
    media_type: str = "application/json"
    assume_type: str | None = None              # the content type to assume if the source sends none (a type that is sent and not accepted is still refused)
    fields: tuple[Field, ...] = ()
    query_help: dict = {}

    @property
    def hosts(self) -> tuple[str, ...]:
        return ()

    @property
    def network(self) -> bool:
        return bool(self.hosts)

    def check_query(self, query: dict) -> dict:
        raise NotImplementedError

    def dataset_id(self, query: dict) -> str:
        raise NotImplementedError

    def url(self, query: dict) -> str:
        raise NotImplementedError

    def parse(self, body: bytes) -> list[dict]:
        raise NotImplementedError

    def verify(self, records: list[dict], query: dict) -> None:
        """Raise ValidationError if the parsed records are not what `query` asked for (default: nothing to check)."""

    def schema(self) -> dict:
        return {"adapter": self.id, "adapter_version": self.version, "fields": [f.to_dict() for f in self.fields]}

    def describe(self) -> dict:
        return {"id": self.id, "source": self.source, "license": self.license, "hosts": list(self.hosts), "query": self.query_help, "network": self.network}


def _json(body: bytes):
    try:
        return json.loads(body.decode("utf-8"), parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValidationError("the response is not valid JSON") from None


class PubChemCompound(Adapter):
    """Basic properties of a compound from PubChem's PUG REST service, by name or by compound id (CID)."""

    id = "pubchem.compound"
    source = "PubChem (NCBI), PUG REST"
    license = "PubChem data are open; individual sources may carry their own terms (https://pubchem.ncbi.nlm.nih.gov/docs/downloads)"
    fields = (Field("cid", "int", minimum=1), Field("formula", "str", pattern=r"[A-Za-z0-9.()+\-\[\]]{1,100}"), Field("molecular_weight", "float", minimum=0.0, maximum=1e6),
              Field("inchikey", "str", pattern=r"[A-Z]{14}-[A-Z]{10}-[A-Z]"), Field("iupac_name", "str", required=False, max_length=1000))
    query_help = {"name": "a compound name, up to 100 characters", "cid": "a compound id, instead of a name"}
    PROPERTIES = "MolecularFormula,MolecularWeight,InChIKey,IUPACName"
    NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,\-()+\[\]']{0,99}")

    def __init__(self, base_url: str = "https://pubchem.ncbi.nlm.nih.gov"):
        self.base_url = base_url.rstrip("/")

    @property
    def hosts(self):
        return (urlsplit(self.base_url).hostname,)

    def check_query(self, query: dict) -> dict:
        if set(query) not in ({"name"}, {"cid"}):
            raise ValidationError("a PubChem query is either {name: ...} or {cid: ...}")
        if "name" in query:
            if not isinstance(query["name"], str) or not self.NAME.fullmatch(query["name"]):
                raise ValidationError("a compound name uses letters, digits, spaces and , - ( ) + [ ] ' only, and is at most 100 characters")
            return {"name": query["name"]}
        if isinstance(query["cid"], bool) or not isinstance(query["cid"], int) or query["cid"] < 1:
            raise ValidationError("a compound id is a positive whole number")
        return {"cid": query["cid"]}

    def dataset_id(self, query: dict) -> str:
        query = self.check_query(query)
        return f"{self.id}.cid-{query['cid']}" if "cid" in query else f"{self.id}.{_slug(query['name'])}"

    def url(self, query: dict) -> str:
        query = self.check_query(query)
        who = f"cid/{query['cid']}" if "cid" in query else f"name/{quote(query['name'], safe='')}"
        return f"{self.base_url}/rest/pug/compound/{who}/property/{self.PROPERTIES}/JSON"

    def parse(self, body: bytes) -> list[dict]:
        document = _json(body)
        try:
            rows = document["PropertyTable"]["Properties"]
            if not isinstance(rows, list):
                raise TypeError
            return [{"cid": row["CID"], "formula": row["MolecularFormula"], "molecular_weight": float(row["MolecularWeight"]), "inchikey": row["InChIKey"],
                     **({"iupac_name": row["IUPACName"]} if "IUPACName" in row else {})} for row in rows]
        except (KeyError, TypeError, ValueError):
            raise ValidationError("the PubChem response does not have the documented PropertyTable shape") from None


class RcsbEntry(Adapter):
    """Summary of a Protein Data Bank entry from RCSB's Data API."""

    id = "rcsb.entry"
    source = "RCSB Protein Data Bank, Data API"
    license = "PDB archive data are available under CC0 1.0 (https://www.rcsb.org/pages/policies)"
    fields = (Field("id", "str", pattern=r"[0-9][A-Z0-9]{3}"), Field("title", "str", max_length=1000), Field("method", "str", required=False, max_length=200),
              Field("resolution_angstrom", "float", required=False, minimum=0.0, maximum=100.0), Field("release_date", "str", required=False, pattern=r"\d{4}-\d{2}-\d{2}"),
              Field("polymer_entity_count", "int", required=False, minimum=0, maximum=100000))
    query_help = {"entry": "a four-character PDB id such as 4HHB"}
    ENTRY = re.compile(r"[0-9][A-Za-z0-9]{3}")

    def __init__(self, base_url: str = "https://data.rcsb.org"):
        self.base_url = base_url.rstrip("/")

    @property
    def hosts(self):
        return (urlsplit(self.base_url).hostname,)

    def check_query(self, query: dict) -> dict:
        if set(query) != {"entry"} or not isinstance(query["entry"], str) or not self.ENTRY.fullmatch(query["entry"]):
            raise ValidationError("an RCSB query is {entry: <four-character PDB id>}")
        return {"entry": query["entry"].upper()}

    def dataset_id(self, query: dict) -> str:
        return f"{self.id}.{self.check_query(query)['entry'].lower()}"

    def url(self, query: dict) -> str:
        return f"{self.base_url}/rest/v1/core/entry/{self.check_query(query)['entry']}"

    def parse(self, body: bytes) -> list[dict]:
        document = _json(body)
        try:
            info = document.get("rcsb_entry_info") or {}
            resolution = info.get("resolution_combined")
            released = (document.get("rcsb_accession_info") or {}).get("initial_release_date")
            methods = document.get("exptl") or []
            record = {"id": document["rcsb_id"], "title": document["struct"]["title"]}
            if methods and isinstance(methods[0], dict) and "method" in methods[0]:
                record["method"] = methods[0]["method"]
            if isinstance(resolution, list) and resolution:
                record["resolution_angstrom"] = float(resolution[0])
            if isinstance(released, str) and len(released) >= 10:
                record["release_date"] = released[:10]
            if "polymer_entity_count" in info:
                record["polymer_entity_count"] = info["polymer_entity_count"]
            return [record]
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ValidationError("the RCSB response does not have the documented entry shape") from None


class NistIonEnergetics(Adapter):
    """The NIST proton affinities and gas-phase acidities that ship with this package (no network)."""

    id = "nist.ion_energetics"
    source = "NIST Chemistry WebBook (SRD 69), as bundled in substrate/data"
    license = "NIST Standard Reference Data: see the notice at https://webbook.nist.gov"
    fields = (Field("id", "str", pattern=r"[a-z0-9_]{1,80}"), Field("quantity", "str", max_length=80), Field("species", "str", max_length=80), Field("value", "float", minimum=0, maximum=1e5),
              Field("uncertainty", "float", required=False, minimum=0, maximum=1e5), Field("unit", "str", max_length=20), Field("method", "str", required=False, max_length=200),
              Field("reference", "str", required=False, max_length=500), Field("comment", "str", required=False, max_length=1000))
    query_help = {"set": "must be empty"}

    def check_query(self, query: dict) -> dict:
        if query:
            raise ValidationError("the bundled NIST set takes no query")
        return {}

    def dataset_id(self, query: dict) -> str:
        return self.id

    def read(self) -> bytes:
        from ...datasets import load_reference_set
        values = load_reference_set("nist_ion_energetics")
        rows = [{"id": v.id, "quantity": v.quantity, "species": v.species, "value": float(v.value), "unit": "kJ/mol", "method": v.method, "reference": v.reference,
                 "comment": v.comment, **({"uncertainty": float(v.uncertainty)} if v.uncertainty is not None else {})} for v in values.values.values()]
        return json.dumps({"source": f"{values.source['database']}, retrieved {values.source['retrieved']}", "records": rows}, sort_keys=True).encode()

    def parse(self, body: bytes) -> list[dict]:
        document = _json(body)
        try:
            return [{k: v for k, v in row.items() if v not in (None, "")} for row in document["records"]]
        except (KeyError, TypeError, AttributeError):
            raise ValidationError("the bundled NIST data has an unexpected shape") from None


class JplSpectralLines(Adapter):
    """The line list of one species from NASA JPL's Submillimeter, Millimeter, and Microwave Spectral Line Catalog, by species tag (water is 18003).

    The file is the catalog's fixed-width text, one 80-character line per spectral line: FREQ F13.4, ERR F8.4, LGINT F8.4, DR I2, ELO F10.4, GUP I3, TAG I7, QNFMT I4, QN'
    6I2, QN'' 6I2 (the catalog's own description, section 2.1 of the documentation at spec.jpl.nasa.gov/ftp/pub/catalog/doc/catintro.pdf). A negative TAG means the line
    frequency was measured in the laboratory (and ERR is then the experimental error); without the sign it was predicted. The quantum numbers are kept as the catalog's text,
    because their coding depends on QNFMT (and a number above 99 is written with a letter): interpreting them is the job of whoever knows the species.
    """

    id = "jpl.spectral_lines"
    source = "NASA JPL Submillimeter, Millimeter, and Microwave Spectral Line Catalog (Pickett et al., J. Quant. Spectrosc. Radiat. Transfer 60, 883-890, 1998)"
    license = "Made available by NASA JPL; the terms and the citation it asks for are at https://spec.jpl.nasa.gov"
    assume_type = "text/plain"                  # the server sends no Content-Type for the .cat files
    fields = (Field("frequency_mhz", "float", minimum=0.0, maximum=1e8), Field("uncertainty_mhz", "float", minimum=0.0, maximum=1e6),
              Field("log10_intensity_nm2_mhz_300k", "float", minimum=-100.0, maximum=20.0), Field("rotational_dof", "int", minimum=0, maximum=3),
              Field("lower_energy_cm1", "float", minimum=0.0, maximum=1e6), Field("upper_degeneracy", "int", minimum=1, maximum=999),
              Field("tag", "int", minimum=1, maximum=999999), Field("measured", "int", minimum=0, maximum=1), Field("qn_format", "int", minimum=0, maximum=9999),
              Field("upper_qn", "str", required=False, pattern=r"[ 0-9A-Za-z+\-]{0,12}", max_length=12),
              Field("lower_qn", "str", required=False, pattern=r"[ 0-9A-Za-z+\-]{0,12}", max_length=12))
    query_help = {"tag": "the species tag: a whole number of up to six digits (the first three are the mass number), such as 18003 for water; see catdir.cat at the source"}
    FLOAT, INT = re.compile(r"\s*-?(?:\d+\.\d*|\.\d+)\s*"), re.compile(r"\s*-?\d+\s*")      # Fortran F format leaves out a zero before the point: ".0005"
    MIN_LINE, MAX_LINE = 55, 80                 # through QNFMT, which is the last field a line must have; the quantum numbers may be cut short

    def __init__(self, base_url: str = "https://spec.jpl.nasa.gov"):
        self.base_url = base_url.rstrip("/")

    @property
    def hosts(self):
        return (urlsplit(self.base_url).hostname,)

    def check_query(self, query: dict) -> dict:
        tag = query.get("tag")
        if set(query) != {"tag"} or isinstance(tag, bool) or not isinstance(tag, int) or not 1 <= tag <= 999999:
            raise ValidationError("a JPL query is {tag: <the species tag, a whole number from 1 to 999999>}")
        return {"tag": tag}

    def dataset_id(self, query: dict) -> str:
        return f"{self.id}.{self.check_query(query)['tag']:06d}"

    def url(self, query: dict) -> str:
        return f"{self.base_url}/ftp/pub/catalog/c{self.check_query(query)['tag']:06d}.cat"

    def parse(self, body: bytes) -> list[dict]:
        try:
            text = body.decode("ascii")
        except UnicodeDecodeError:
            raise ValidationError("a JPL catalog file is ASCII text") from None
        lines = [line for line in text.replace("\r", "").split("\n") if line.strip()]
        if not lines:
            raise ValidationError("the JPL catalog file has no lines")
        if len(lines) > MAX_RECORDS:
            raise ValidationError(f"the JPL catalog file has {len(lines):,} lines, more than the {MAX_RECORDS:,} accepted")
        records = []
        for number, line in enumerate(lines, 1):
            if not self.MIN_LINE <= len(line) <= self.MAX_LINE:
                raise ValidationError(f"line {number} of the JPL catalog file is not a catalog line (it has {len(line)} characters)")
            cells = (line[0:13], line[13:21], line[21:29], line[29:31], line[31:41], line[41:44], line[44:51], line[51:55])
            floats, ints = (cells[0], cells[1], cells[2], cells[4]), (cells[3], cells[5], cells[6], cells[7])
            if not all(self.FLOAT.fullmatch(c) for c in floats) or not all(self.INT.fullmatch(c) for c in ints):
                raise ValidationError(f"line {number} of the JPL catalog file does not have the catalog's fixed-width layout")
            tag = int(cells[6])
            records.append({"frequency_mhz": float(cells[0]), "uncertainty_mhz": float(cells[1]), "log10_intensity_nm2_mhz_300k": float(cells[2]),
                            "rotational_dof": int(cells[3]), "lower_energy_cm1": float(cells[4]), "upper_degeneracy": int(cells[5]), "tag": abs(tag),
                            "measured": 1 if tag < 0 else 0, "qn_format": int(cells[7]), "upper_qn": line[55:67].rstrip(), "lower_qn": line[67:79].rstrip()})
        return records

    def verify(self, records: list[dict], query: dict) -> None:
        wanted = self.check_query(query)["tag"]
        other = sorted({r["tag"] for r in records} - {wanted})
        if other:
            raise ValidationError(f"the file holds lines of species {other[0]}, not the species {wanted} that was asked for")


def default_adapters() -> dict[str, Adapter]:
    adapters = [NistIonEnergetics(), PubChemCompound(), RcsbEntry(), JplSpectralLines()]
    return {a.id: a for a in adapters}
