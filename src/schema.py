from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryBundle:
    raw_query: str
    intent: str
    rewritten_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    search_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    chunk_id: str
    section: str | None
    page: int | None
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperCandidate:
    paper_id: str
    title: str
    authors: list[str] | None
    year: int | None
    venue: str | None
    score: float
    match_reason: list[str] = field(default_factory=list)
    evidences: list[EvidenceRecord] = field(default_factory=list)
    rel_path: str | None = None
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidences"] = [evidence.to_dict() for evidence in self.evidences]
        return data


@dataclass
class AnswerPayload:
    answer_text: str
    papers: list[PaperCandidate]
    confidence: float
    status: str
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "papers": [paper.to_dict() for paper in self.papers],
            "confidence": self.confidence,
            "status": self.status,
            "debug": self.debug,
        }
