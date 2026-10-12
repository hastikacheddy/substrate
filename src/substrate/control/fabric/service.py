"""Ingesting a dataset: fetch (or read), keep the raw bytes, check the records, store a new immutable version if anything changed, and write down where it came from.

An experiment pins the dataset versions it used, so what it saw cannot change under it: fetching the same query again either finds identical records (and returns the
version already stored) or creates the next version, and the old one stays exactly as it was. Every ingest is a quota-limited, audited action.
"""
from __future__ import annotations

from ...errors import ValidationError
from ..artifacts import Artifacts
from ..registry import Registry
from ..security import Action, Principal
from .adapters import Adapter, default_adapters, validate_records
from .client import FetchError, SafeHTTPClient, redact_url


class Fabric:
    def __init__(self, registry: Registry, artifacts: Artifacts, client: SafeHTTPClient | None = None, adapters: dict[str, Adapter] | None = None):
        self.registry, self.artifacts = registry, artifacts
        self.adapters = default_adapters() if adapters is None else adapters
        self.client = client or SafeHTTPClient({h for a in self.adapters.values() for h in a.hosts if h})

    def describe(self) -> list[dict]:
        return [a.describe() for a in self.adapters.values()]

    def ingest(self, principal: Principal, adapter_id: str, query: dict | None = None, headers: dict[str, str] | None = None) -> dict:
        principal.require(Action.DATASET_INGEST)
        adapter = self.adapters.get(adapter_id)
        if adapter is None:
            raise ValidationError(f"no data adapter '{adapter_id}' (have: {', '.join(self.adapters)})")
        query = adapter.check_query(dict(query or {}))                      # refused before an allowance is spent or a request is made
        dataset_id = adapter.dataset_id(query)
        self.registry.consume(principal, "dataset_ingests_per_day", 1)
        if adapter.network:
            url = adapter.url(query)
            response = self.client.fetch(url, headers=headers)
            body, content_type, source_url, retrieved_at = response.body, response.content_type, response.url, response.retrieved_at
        else:
            body, content_type, source_url, retrieved_at = adapter.read(), "application/json", f"bundled:{adapter.id}", self.registry.clock()
        records = adapter.parse(body)
        validate_records(records, adapter.fields, owner=dataset_id)             # nothing is kept of a response that does not check out
        media_type = content_type if content_type in ("application/json", "text/csv", "text/plain") else "text/plain"
        raw = self.artifacts.write(principal.tenant_id, body, name=f"{dataset_id}.raw", media_type=media_type)
        provenance = {"adapter": adapter.id, "adapter_version": adapter.version, "query": query, "url": redact_url(source_url) if adapter.network else source_url,
                      "retrieved_at": retrieved_at, "content_type": content_type, "raw_bytes": len(body), "ingested_by": principal.id, "tenant": principal.tenant_id}
        dataset = self.registry.put_dataset(dataset_id, source=adapter.source, license=adapter.license, schema=adapter.schema(), records=records, provenance=provenance,
                                            raw_sha256=raw.sha256)
        self.registry.audit(principal.tenant_id, "dataset.ingest", by=principal.id, dataset=dataset_id, version=dataset["version"], checksum=dataset["checksum"],
                            adapter=adapter.id, url=provenance["url"])
        return dataset


__all__ = ["Fabric", "FetchError"]
