from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobFilters(BaseModel):
    company: str | None = None
    location: str | None = None
    industry: str | None = None
    job_type: str | None = None


class PortfolioIngestRequest(BaseModel):
    portfolio_id: str = Field(min_length=1)
    student_id: str = Field(min_length=1)
    title: str = ""
    description: str = Field(min_length=1)
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobIngestRequest(BaseModel):
    job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = ""
    description: str = Field(min_length=1)
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    industry: str = ""
    location: str = ""
    job_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchRequest(BaseModel):
    portfolio_text: str = Field(min_length=1)
    portfolio_id: str | None = None
    student_id: str | None = None
    title: str = ""
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    filters: JobFilters = Field(default_factory=JobFilters)
    top_k: int = Field(default=25, ge=1, le=100)
    use_reranker: bool = True


class MatchResult(BaseModel):
    job_id: str
    title: str
    company: str = ""
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    id: str


class MatchResponse(BaseModel):
    status: str
    query_text: str
    total_candidates: int
    results: list[MatchResult]
