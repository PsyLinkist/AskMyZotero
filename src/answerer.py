"""本文件负责把聚合后的论文结果整理成最终回答内容和返回结构。"""

from __future__ import annotations

from typing import Any

from src.domain_models import AnswerPayload, PaperCandidate


def _get_bundle_value(query_bundle: Any, key: str, default: Any) -> Any:
    if hasattr(query_bundle, key):
        return getattr(query_bundle, key)
    return query_bundle.get(key, default)


def _format_paper_line(index: int, paper: PaperCandidate) -> str:
    meta_parts: list[str] = []
    if paper.year:
        meta_parts.append(str(paper.year))
    if paper.venue:
        meta_parts.append(str(paper.venue))
    meta_text = f" ({', '.join(meta_parts)})" if meta_parts else ""

    lines = [f"{index}. {paper.title}{meta_text}"]
    if paper.match_reason:
        lines.append(f"   - 理由：{'；'.join(paper.match_reason[:2])}")
    if paper.evidences:
        evidence = paper.evidences[0]
        page_text = f"第 {evidence.page} 页" if evidence.page else "页码未知"
        snippet = evidence.text.replace("\n", " ").strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        lines.append(f"   - 证据：{page_text}，{snippet}")
    return "\n".join(lines)


def generate_answer(query_bundle: Any, papers: list[PaperCandidate]) -> AnswerPayload:
    raw_query = _get_bundle_value(query_bundle, "raw_query", "")
    rewritten_queries = _get_bundle_value(query_bundle, "rewritten_queries", [])
    intent = _get_bundle_value(query_bundle, "intent", "paper_search")

    if not papers:
        return AnswerPayload(
            answer_text="当前检索结果不足以支撑论文级回答。",
            papers=[],
            confidence=0.0,
            status="retrieval_failed",
            debug={
                "intent": intent,
                "rewritten_queries": rewritten_queries,
                "retrieved_chunk_count": 0,
                "retrieved_paper_count": 0,
            },
        )

    total_evidence_count = sum(len(paper.evidences) for paper in papers)
    confidence = min(1.0, max(0.2, papers[0].score / 3.0))
    status = "ok" if total_evidence_count > 0 else "insufficient_evidence"

    lines = [f"问题：{raw_query}", "", "回答："]
    for index, paper in enumerate(papers, start=1):
        lines.append(_format_paper_line(index, paper))
    if status != "ok":
        lines.append("")
        lines.append("当前结果存在证据不足的风险，建议进一步缩小问题范围或查看原文。")

    return AnswerPayload(
        answer_text="\n".join(lines).strip(),
        papers=papers,
        confidence=confidence,
        status=status,
        debug={
            "intent": intent,
            "rewritten_queries": rewritten_queries,
            "retrieved_chunk_count": total_evidence_count,
            "retrieved_paper_count": len(papers),
        },
    )
