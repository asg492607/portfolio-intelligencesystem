from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb

from matchmaking.config import Settings


@dataclass(slots=True)
class CollectionRecord:
    record_id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]


class ChromaRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = self._build_client(settings)
        self.portfolios = self.client.get_or_create_collection(
            name=settings.portfolio_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.jobs = self.client.get_or_create_collection(
            name=settings.job_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_client(settings: Settings):
        if settings.chroma_mode == "local":
            return chromadb.PersistentClient(path=str(settings.chroma_path))

        cloud_kwargs: dict[str, object] = {}
        if settings.chroma_api_key:
            cloud_kwargs["api_key"] = settings.chroma_api_key
        if settings.chroma_tenant:
            cloud_kwargs["tenant"] = settings.chroma_tenant
        if settings.chroma_database:
            cloud_kwargs["database"] = settings.chroma_database
        if settings.chroma_host:
            cloud_kwargs["cloud_host"] = settings.chroma_host
            cloud_kwargs["cloud_port"] = settings.chroma_port
            cloud_kwargs["enable_ssl"] = settings.chroma_enable_ssl
        return chromadb.CloudClient(**cloud_kwargs)

    @staticmethod
    def _upsert(collection, record: CollectionRecord) -> None:
        collection.upsert(
            ids=[record.record_id],
            documents=[record.document],
            metadatas=[record.metadata],
            embeddings=[record.embedding],
        )

    def upsert_portfolio(self, record: CollectionRecord) -> None:
        self._upsert(self.portfolios, record)

    def upsert_job(self, record: CollectionRecord) -> None:
        self._upsert(self.jobs, record)

    def query_jobs(
        self,
        *,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_args: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        return self.jobs.query(**query_args)

    def fetch_all_jobs(self) -> dict[str, Any]:
        return self.jobs.get(include=["documents", "metadatas"])
