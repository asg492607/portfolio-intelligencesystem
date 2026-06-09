from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from matchmaking.config import Settings
from matchmaking.schemas import JobFilters, JobIngestRequest, MatchRequest, MatchResult, PortfolioIngestRequest
from matchmaking.services.embedding import EmbeddingService
from matchmaking.services.store import ChromaRepository, CollectionRecord
from matchmaking.services.text_utils import (
    build_job_search_text,
    build_portfolio_search_text,
    safe_json,
    tokenize,
)


@dataclass(slots=True)
class CachedJob:
    job_id: str
    document: str
    metadata: dict[str, Any]


class HybridSearchEngine:
    def __init__(self, settings: Settings, repo: ChromaRepository, embedder: EmbeddingService) -> None:
        self.settings = settings
        self.repo = repo
        self.embedder = embedder
        self.cached_jobs: list[CachedJob] = []
        self.job_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.reranker: CrossEncoder | None = None

        if settings.enable_reranker:
            try:
                self.reranker = CrossEncoder(settings.reranker_model_name)
            except Exception:
                self.reranker = None

        self.refresh_job_cache()

    def refresh_job_cache(self) -> None:
        raw = self.repo.fetch_all_jobs()
        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        self.cached_jobs = []
        self.job_tokens = []

        for job_id, document, metadata in zip(ids, documents, metadatas):
            self.cached_jobs.append(
                CachedJob(
                    job_id=job_id,
                    document=document or "",
                    metadata=metadata or {},
                )
            )
            self.job_tokens.append(tokenize(document or ""))

        self.bm25 = BM25Okapi(self.job_tokens) if self.job_tokens else None

    def ingest_portfolio(self, payload: PortfolioIngestRequest) -> str:
        document = build_portfolio_search_text(
            title=payload.title,
            description=payload.description,
            skills=payload.skills,
            tools=payload.tools,
            target_roles=payload.target_roles,
            industries=payload.industries,
        )
        embedding = self.embedder.embed_query(document)
        metadata = {
            "record_type": "portfolio",
            "portfolio_id": payload.portfolio_id,
            "student_id": payload.student_id,
            "title": payload.title,
            "description": payload.description,
            "skills_text": ", ".join(payload.skills),
            "tools_text": ", ".join(payload.tools),
            "target_roles_text": ", ".join(payload.target_roles),
            "industries_text": ", ".join(payload.industries),
            "raw_json": safe_json(payload.model_dump()),
        }
        metadata.update({k: v for k, v in payload.metadata.items() if isinstance(v, (str, int, float, bool))})
        self.repo.upsert_portfolio(
            CollectionRecord(
                record_id=payload.portfolio_id,
                document=document,
                metadata=metadata,
                embedding=embedding,
            )
        )
        return payload.portfolio_id

    def ingest_job(self, payload: JobIngestRequest) -> str:
        document = build_job_search_text(
            title=payload.title,
            company=payload.company,
            description=payload.description,
            skills=payload.skills,
            tools=payload.tools,
            industry=payload.industry,
            location=payload.location,
            job_type=payload.job_type,
        )
        embedding = self.embedder.embed_query(document)
        metadata = {
            "record_type": "job",
            "job_id": payload.job_id,
            "title": payload.title,
            "company": payload.company,
            "description": payload.description,
            "skills_text": ", ".join(payload.skills),
            "tools_text": ", ".join(payload.tools),
            "industry": payload.industry,
            "location": payload.location,
            "job_type": payload.job_type,
            "raw_json": safe_json(payload.model_dump()),
        }
        metadata.update({k: v for k, v in payload.metadata.items() if isinstance(v, (str, int, float, bool))})
        self.repo.upsert_job(
            CollectionRecord(
                record_id=payload.job_id,
                document=document,
                metadata=metadata,
                embedding=embedding,
            )
        )
        self.refresh_job_cache()
        return payload.job_id

    def _filters_to_where(self, filters: JobFilters) -> dict[str, Any] | None:
        clauses: dict[str, Any] = {}
        for key in ("company", "location", "industry", "job_type"):
            value = getattr(filters, key)
            if value:
                clauses[key] = value
        return clauses or None

    def _semantic_candidates(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: JobFilters,
    ) -> dict[str, dict[str, Any]]:
        result = self.repo.query_jobs(
            query_embedding=query_embedding,
            n_results=top_k,
            where=self._filters_to_where(filters),
        )
        candidates: dict[str, dict[str, Any]] = {}
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for job_id, document, metadata, distance in zip(ids, docs, metas, distances):
            semantic_score = max(0.0, 1.0 - float(distance or 0.0))
            candidates[job_id] = {
                "job_id": job_id,
                "document": document or "",
                "metadata": metadata or {},
                "semantic_score": semantic_score,
                "lexical_score": 0.0,
                "rerank_score": None,
                "final_score": semantic_score,
            }
        return candidates

    def _lexical_scores(self, query_text: str, filters: JobFilters) -> dict[str, float]:
        if not self.cached_jobs or self.bm25 is None:
            return {}

        query_tokens = tokenize(query_text)
        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.size == 0:
            return {}

        max_score = float(scores.max()) if float(scores.max()) > 0 else 1.0
        order = scores.argsort()[::-1][: self.settings.bm25_top_k]
        query_filters = self._filters_to_where(filters)

        result: dict[str, float] = {}
        for index in order:
            job = self.cached_jobs[int(index)]
            if query_filters:
                metadata = job.metadata
                if any(str(metadata.get(field, "")) != str(value) for field, value in query_filters.items()):
                    continue
            result[job.job_id] = float(scores[int(index)] / max_score)
        return result

    def _rerank(self, query_text: str, candidates: list[MatchResult]) -> list[MatchResult]:
        if not self.reranker or not candidates:
            return candidates

        pairs = []
        for candidate in candidates:
            metadata = candidate.metadata
            doc = metadata.get("description", "") or candidate.metadata.get("raw_json", "")
            pairs.append((query_text, f"{candidate.title} {candidate.company} {doc}"))

        rerank_scores = self.reranker.predict(pairs)
        if not isinstance(rerank_scores, list):
            rerank_scores = list(rerank_scores)

        scored: list[MatchResult] = []
        for candidate, score in zip(candidates, rerank_scores):
            candidate.rerank_score = float(score)
            candidate.final_score = 0.85 * float(score) + 0.15 * candidate.final_score
            scored.append(candidate)
        return sorted(scored, key=lambda item: item.final_score, reverse=True)

    def match(self, payload: MatchRequest) -> tuple[str, list[MatchResult]]:
        query_text = build_portfolio_search_text(
            title=payload.title,
            description=payload.portfolio_text,
            skills=payload.skills,
            tools=payload.tools,
            target_roles=payload.target_roles,
            industries=payload.industries,
        )
        query_embedding = self.embedder.embed_query(query_text)

        semantic_candidates = self._semantic_candidates(
            query_embedding=query_embedding,
            top_k=self.settings.semantic_top_k,
            filters=payload.filters,
        )
        lexical_scores = self._lexical_scores(query_text, payload.filters)

        merged: dict[str, MatchResult] = {}

        for job_id, data in semantic_candidates.items():
            merged[job_id] = MatchResult(
                job_id=job_id,
                title=str(data["metadata"].get("title", "")),
                company=str(data["metadata"].get("company", "")),
                semantic_score=float(data["semantic_score"]),
                lexical_score=0.0,
                rerank_score=None,
                final_score=float(data["semantic_score"]),
                metadata=dict(data["metadata"]),
            )

        for job_id, lexical_score in lexical_scores.items():
            if job_id in merged:
                merged[job_id].lexical_score = float(lexical_score)
                merged[job_id].final_score = 0.65 * merged[job_id].semantic_score + 0.35 * float(lexical_score)
            else:
                job = next((item for item in self.cached_jobs if item.job_id == job_id), None)
                if job is None:
                    continue
                metadata = dict(job.metadata)
                merged[job_id] = MatchResult(
                    job_id=job_id,
                    title=str(metadata.get("title", "")),
                    company=str(metadata.get("company", "")),
                    semantic_score=0.0,
                    lexical_score=float(lexical_score),
                    rerank_score=None,
                    final_score=float(lexical_score),
                    metadata=metadata,
                )

        candidates = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)[: self.settings.rerank_top_k]
        if payload.use_reranker and self.settings.enable_reranker:
            candidates = self._rerank(query_text, candidates)

        return query_text, candidates[: payload.top_k]
