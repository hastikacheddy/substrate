"""Adapters: how one outside dataset is asked for, read, and checked.

An adapter knows four things about its source: how to build the request URL from a query (and to refuse a query that is not what the source accepts), which hosts it
needs, how to turn the response into flat records, and what every record must look like. Everything that comes back is untrusted: the records are checked against the
adapter's declared fields (type, range, pattern, length, finiteness) and an unexpected field or a malformed value is an error, not something quietly passed on.

The adapters here were written from the sources' published API documentation and are exercised in the tests against responses of that documented shape; they were not
run against the live services when they were written. `tests/test_control_fabric_*` says what is and is not checked.
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


def default_adapters() -> dict[str, Adapter]:
    adapters = [NistIonEnergetics(), PubChemCompound(), RcsbEntry()]
    return {a.id: a for a in adapters}
