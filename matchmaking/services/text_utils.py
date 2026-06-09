from __future__ import annotations

import json
import re
from typing import Any


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def join_nonempty(values: list[str]) -> str:
    return ", ".join(item.strip() for item in values if item and item.strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def build_portfolio_search_text(
    *,
    title: str = "",
    description: str,
    skills: list[str],
    tools: list[str],
    target_roles: list[str],
    industries: list[str],
) -> str:
    parts = [
        normalize_text(title),
        normalize_text(description),
        f"Skills: {join_nonempty(skills)}" if skills else "",
        f"Tools: {join_nonempty(tools)}" if tools else "",
        f"Target roles: {join_nonempty(target_roles)}" if target_roles else "",
        f"Industries: {join_nonempty(industries)}" if industries else "",
    ]
    return normalize_text(" | ".join(part for part in parts if part))


def build_job_search_text(
    *,
    title: str,
    company: str,
    description: str,
    skills: list[str],
    tools: list[str],
    industry: str,
    location: str,
    job_type: str,
) -> str:
    parts = [
        f"Role: {normalize_text(title)}" if title else "",
        f"Company: {normalize_text(company)}" if company else "",
        f"Industry: {normalize_text(industry)}" if industry else "",
        f"Location: {normalize_text(location)}" if location else "",
        f"Type: {normalize_text(job_type)}" if job_type else "",
        normalize_text(description),
        f"Required skills: {join_nonempty(skills)}" if skills else "",
        f"Tools: {join_nonempty(tools)}" if tools else "",
    ]
    return normalize_text(" | ".join(part for part in parts if part))
