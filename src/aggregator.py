"""本文件负责将检索到的文本片段聚合成论文级结果，并进行简单打分。"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.domain_models import AnswerPayload, EvidenceRecord, PaperCandidate


BASE_SECTION_WEIGHTS = {
    "abstract": 1.30,
    "method": 1.40,
    "experiment": 1.20,
    "conclusion": 1.10,
    "introduction": 0.95,
    "body": 1.00,
    "references": 0.35,
}

INTENT_SECTION_OVERRIDES = {
    "fact_qa": {"method": 1.60, "abstract": 1.35, "experiment": 1.25},
    "definition": {"abstract": 1.50, "introduction": 1.25, "method": 1.10},
    "comparison": {"method": 1.50, "experiment": 1.40, "conclusion": 1.20},
    "survey": {"abstract": 1.45, "conclusion": 1.25, "introduction": 1.15},
    "paper_lookup": {"abstract": 1.35, "method": 1.35, "references": 0.50},
}


def _get_bundle_value(query_bundle: Any, key: str, default: Any = None) -> Any:
    if hasattr(query_bundle, key):
        return getattr(query_bundle, key)
    if isinstance(query_bundle, dict):
        return query_bundle.get(key, default)
    return default


def resolve_section_weights(intent: str | None) -> dict[str, float]:
    weights = dict(BASE_SECTION_WEIGHTS)
    if intent:
        weights.update(INTENT_SECTION_OVERRIDES.get(str(intent), {}))
    return weights


def _normalize_query_terms(query_bundle: Any) -> list[str]:
    raw_query = getattr(query_bundle, "raw_query", None)
    if raw_query is None:
        raw_query = str(query_bundle.get("raw_query", ""))
    return [term.lower() for term in str(raw_query).split() if len(term.strip()) >= 2]


def _normalize_text(value: Any, max_len: int = 300) -> str:
    text = " ".join(str(value or "").lower().split())
    return text[:max_len]


def _split_into_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?;。！？；])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _build_evidence_snippet(text: str, query_terms: list[str], max_chars: int = 280) -> str:
    sentences = _split_into_sentences(text)
    if not sentences:
        return ""

    matched = [
        sentence for sentence in sentences
        if any(term in sentence.lower() for term in query_terms)
    ]
    chosen = matched[:2] if matched else sentences[:2]
    snippet = " ".join(chosen).strip()
    if len(snippet) > max_chars:
        return snippet[: max_chars - 3].rstrip() + "..."
    return snippet


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
    intent = _get_bundle_value(query_bundle, "intent", "paper_lookup")
    section_weights = resolve_section_weights(intent)

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
        base_score = float(chunk.get("score", 0.0))
        weighted_score = base_score * section_weights.get(chunk_type, 1.0)
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
                    text=_build_evidence_snippet(text, query_terms),
                    score=weighted_score,
                )
            )

    papers: list[PaperCandidate] = []
    for paper in grouped.values():
        evidences = sorted(paper["evidences"], key=lambda item: item.score, reverse=True)[:2]
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


def _build_paper_context(index: int, paper: PaperCandidate) -> str:
    parts = [f"[{index}]"]
    if paper.match_reason:
        parts.append(f"匹配原因：{'；'.join(paper.match_reason[:3])}")

    evidence_lines: list[str] = []
    for evidence in paper.evidences[:2]:
        page_text = f"第 {evidence.page} 页" if evidence.page else "页码未知"
        section_text = evidence.section or "正文"
        snippet = evidence.text.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        evidence_lines.append(f"- {page_text} · {section_text}：{snippet}")

    if evidence_lines:
        parts.append("证据：")
        parts.extend(evidence_lines)
    return "\n".join(parts)


def _fallback_answer_text(query_bundle: Any, papers: list[PaperCandidate]) -> str:
    intent = _get_bundle_value(query_bundle, "intent", "paper_lookup")
    citations = "".join(f"[{index}]" for index, _ in enumerate(papers[:5], start=1))
    if intent == "survey":
        return f"与该主题最相关的论文主要见 {citations}。详细题名、作者、页码和证据摘要请查看右侧 SOURCES。"
    return f"与该问题最相关的论文见 {citations}。详细题名、作者、页码和证据摘要请查看右侧 SOURCES。"


def _sanitize_answer_text(answer_text: str, paper_count: int) -> str:
    if not answer_text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in str(answer_text).splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        lower_line = line.lower()
        if re.match(r"^\d+\.\s+", line):
            continue
        if re.match(r"^[-*]\s*(理由|证据|参考文献|来源)\s*[：:]", line):
            continue
        if re.match(r"^(理由|证据|参考文献|来源)\s*[：:]", line):
            continue
        if any(keyword in lower_line for keyword in ("authors:", "author:", "venue:", "year:", "title:")):
            continue

        line = re.sub(r"\[\s*(\d+)\s*\]", lambda m: f"[{m.group(1)}]" if 1 <= int(m.group(1)) <= paper_count else "", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not re.search(r"\[(\d+)\]", cleaned):
        suffix = "".join(f"[{index}]" for index in range(1, min(paper_count, 5) + 1))
        cleaned = f"{cleaned} {suffix}".strip()
    return cleaned


def _generate_llm_answer_text(query_bundle: Any, papers: list[PaperCandidate], llm: Any) -> tuple[str, dict[str, Any]]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个学术文献助手。请基于给定论文证据回答用户问题。"
                "回答应自然、简洁、有信息量。"
                "如需引用来源，只能使用编号引用，如 [1]、[2]。"
                "不要输出论文标题、作者、年份、期刊、页码。"
                "不要输出“理由：”“证据：”“参考文献：”“来源：”等列表。"
                "详细文献信息由界面右侧 SOURCES 面板展示，你只负责正文回答。"
            ),
            (
                "human",
                "用户问题：{query}\n\n"
                "候选论文证据：\n{paper_context}\n\n"
                "请直接给出回答正文，只使用 [n] 形式引用。",
            ),
        ]
    )
    paper_context = "\n\n".join(
        _build_paper_context(index, paper) for index, paper in enumerate(papers, start=1)
    )
    messages = prompt.format_messages(
        query=_get_bundle_value(query_bundle, "raw_query", ""),
        paper_context=paper_context,
    )
    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke(
        {
            "query": _get_bundle_value(query_bundle, "raw_query", ""),
            "paper_context": paper_context,
        }
    ).strip()
    return raw_response, {
        "prompt_messages": [{"type": getattr(message, "type", "unknown"), "content": str(getattr(message, "content", ""))} for message in messages],
        "raw_response": raw_response,
    }


def generate_answer(query_bundle: Any, papers: list[PaperCandidate], llm: Any | None = None) -> AnswerPayload:
    intent = _get_bundle_value(query_bundle, "intent", "paper_lookup")
    section_weights = resolve_section_weights(intent)
    raw_query = _get_bundle_value(query_bundle, "raw_query", "")

    if not papers:
        return AnswerPayload(
            answer_text="当前没有检索到足够相关的论文证据。",
            papers=[],
            confidence=0.0,
            status="empty",
            answer_type="paper_list",
            evidence_summary=[],
            debug={"intent": intent, "section_weights": section_weights},
        )

    evidence_summary: list[str] = []
    for paper in papers[:3]:
        if paper.evidences:
            top_evidence = paper.evidences[0]
            page_text = f"第 {top_evidence.page} 页" if top_evidence.page else "未标页码"
            evidence_summary.append(f"{paper.title}：{page_text} 命中 {top_evidence.section or '正文'} 证据")

    answer_debug: dict[str, Any] = {"mode": "fallback"}
    answer_text = _fallback_answer_text(query_bundle, papers)
    if llm is not None:
        try:
            raw_answer_text, llm_debug = _generate_llm_answer_text(query_bundle, papers, llm)
            answer_text = _sanitize_answer_text(raw_answer_text, len(papers))
            answer_debug = {"mode": "llm", "llm_answer": llm_debug}
        except Exception as exc:
            answer_debug = {"mode": "fallback", "error": str(exc)}
            answer_text = _fallback_answer_text(query_bundle, papers)

    confidence = min(0.95, 0.5 + 0.08 * len(papers))
    return AnswerPayload(
        answer_text=answer_text,
        papers=papers,
        confidence=confidence,
        status="ok",
        answer_type="paper_list",
        evidence_summary=evidence_summary,
        debug={"intent": intent, "section_weights": section_weights, **answer_debug},
    )
