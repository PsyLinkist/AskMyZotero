from __future__ import annotations

from typing import Any

from src.domain_models import EvidenceRecord, PaperCandidate


SECTION_WEIGHT = {
    "abstract": 1.35,
    "introduction": 1.15,
    "method": 1.3,
    "experiment": 1.1,
    "conclusion": 1.1,
    "body": 1.0,
}


def _normalize_query_terms(query_bundle: Any) -> list[str]:
    raw_query = getattr(query_bundle, "raw_query", None)
    if raw_query is None:
        raw_query = str(query_bundle.get("raw_query", ""))
    return [term.lower() for term in str(raw_query).split() if len(term.strip()) >= 2]


def _normalize_text(value: Any, max_len: int = 300) -> str:
    text = " ".join(str(value or "").lower().split())
    return text[:max_len]


def _build_paper_key(chunk: dict[str, Any], fallback_rank: int) -> str:
    paper_id = str(chunk.get("paper_id") or "").strip().lower()
    source_path = str(chunk.get("source_path") or "").strip().lower()
    rel_path = str(chunk.get("rel_path") or "").strip().lower()
    title = _normalize_text(chunk.get("paper_title") or chunk.get("title") or "", max_len=180)

    if paper_id:
        return f"paper::{paper_id}"
    if source_path:
        return f"source::{source_path}"
    if rel_path:
        return f"rel::{rel_path}"
    if title:
        return f"title::{title}"
    return f"paper::{fallback_rank}"


def _build_chunk_key(chunk: dict[str, Any], paper_key: str, fallback_rank: int) -> str:
    chunk_id = str(chunk.get("chunk_id") or "").strip().lower()
    if chunk_id:
        return f"chunk::{chunk_id}"

    page = chunk.get("page_start") or chunk.get("page") or "na"
    snippet = _normalize_text(chunk.get("text") or chunk.get("content") or "")
    return f"{paper_key}::page::{page}::text::{snippet or fallback_rank}"


def aggregate_to_papers(query_bundle: Any, chunks: list[dict[str, Any]], top_papers: int = 5) -> list[PaperCandidate]:
    grouped: dict[str, dict[str, Any]] = {}
    query_terms = _normalize_query_terms(query_bundle)
    seen_chunk_keys: set[str] = set()

    for rank, chunk in enumerate(chunks, start=1):
        paper_key = _build_paper_key(chunk, rank)
        chunk_key = _build_chunk_key(chunk, paper_key, rank)
        if chunk_key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(chunk_key)

        paper_id = str(chunk.get("paper_id") or chunk.get("rel_path") or chunk.get("source_path") or paper_key)
        section = chunk.get("section")
        chunk_type = str(chunk.get("chunk_type") or "body").lower()
        text = str(chunk.get("text") or "")
        title = str(chunk.get("paper_title") or chunk.get("file_name") or paper_id)

        if paper_key not in grouped:
            grouped[paper_key] = {
                "paper_id": paper_id,
                "title": title,
                "authors": chunk.get("authors"),
                "year": chunk.get("year"),
                "venue": chunk.get("venue"),
                "rel_path": chunk.get("rel_path"),
                "source_path": chunk.get("source_path") or chunk.get("source"),
                "score": 0.0,
                "match_reason": set(),
                "evidences": [],
                "seen_evidences": set(),
            }

        paper = grouped[paper_key]
        base_score = float(chunk.get("score", 1.0 / (rank + 1)))
        weighted_score = base_score * SECTION_WEIGHT.get(chunk_type, 1.0)
        paper["score"] += weighted_score

        text_lower = text.lower()
        title_lower = title.lower()
        matched_terms = [term for term in query_terms if term in text_lower or term in title_lower]

        if matched_terms:
            paper["match_reason"].add(f"命中关键词：{', '.join(sorted(set(matched_terms))[:4])}")
        if any(term in title_lower for term in query_terms):
            paper["score"] += 0.25
            paper["match_reason"].add("标题命中查询关键词")
        if section:
            paper["match_reason"].add(f"命中 {section} 段落")
        if chunk_type in {"abstract", "method"}:
            paper["match_reason"].add(f"{chunk_type} 区域证据较强")

        evidence_page = chunk.get("page_start") or chunk.get("page")
        evidence_key = f"{evidence_page}::{_normalize_text(text)}"
        if evidence_key not in paper["seen_evidences"]:
            paper["seen_evidences"].add(evidence_key)
            paper["evidences"].append(
                EvidenceRecord(
                    chunk_id=str(chunk.get("chunk_id") or f"{paper_id}#chunk-{rank}"),
                    section=section,
                    page=evidence_page,
                    text=text.strip(),
                    score=weighted_score,
                )
            )

    papers: list[PaperCandidate] = []
    for paper in grouped.values():
        evidences = sorted(paper["evidences"], key=lambda item: item.score, reverse=True)[:3]
        match_reason = list(sorted(paper["match_reason"])) or ["检索到多条相关正文证据"]
        papers.append(
            PaperCandidate(
                paper_id=paper["paper_id"],
                title=paper["title"],
                authors=paper["authors"],
                year=paper["year"],
                venue=paper["venue"],
                score=paper["score"],
                match_reason=match_reason,
                evidences=evidences,
                rel_path=paper["rel_path"],
                source_path=paper["source_path"],
            )
        )

    papers.sort(key=lambda item: item.score, reverse=True)
    return papers[:top_papers]
