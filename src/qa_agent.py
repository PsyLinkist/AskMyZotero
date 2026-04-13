"""Current Zotero QA agent with intent routing and LLM-assisted query rewrite."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.aggregator import aggregate_to_papers, generate_answer, resolve_section_weights
from src.config import AppConfig
from src.domain_models import QueryBundle
from src.indexer import build_llm, get_vectorstore
from src.manifest import prepare_manifest_snapshot
from src.prompt_logger import save_prompt_log

INTENT_PATTERNS = [
    ("comparison", ("区别", "差异", "对比", "compare", "versus", "vs")),
    ("definition", ("什么是", "是什么", "定义", "meaning of", "what is")),
    ("survey", ("综述", "总结", "梳理", "概览", "overview", "survey")),
    ("paper_lookup", ("论文", "paper", "文献", "article", "哪篇", "哪些论文", "推荐阅读")),
]

FACT_PREFIXES = ("是否", "有没有", "能否", "是不是", "有无")
FACT_HINTS = ("吗", "么", "?", "？", "是否", "does", "is ", "are ", "did ", "can ")


class ZoteroAgent:
    def __init__(self, config: AppConfig):
        print("Initializing Zotero Agent...")
        self.config = config
        prepare_manifest_snapshot(self.config)
        self.vectorstore = get_vectorstore(self.config)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.top_k})
        self.llm = build_llm(self.config)

    def parse_query(self, query: str) -> QueryBundle:
        clean_query = " ".join(query.split())
        lowered = clean_query.lower()
        entities = self._extract_entities(clean_query)
        keywords = self._extract_keywords(clean_query)

        intent = "fact_qa"
        confidence = 0.55
        for candidate, patterns in INTENT_PATTERNS:
            if any(pattern.lower() in lowered for pattern in patterns):
                intent = candidate
                confidence = 0.85
                break
        if clean_query.startswith(FACT_PREFIXES) or any(hint in lowered for hint in FACT_HINTS):
            intent = "fact_qa"
            confidence = max(confidence, 0.8)

        rewritten_queries = self._build_rewritten_queries(clean_query)
        return QueryBundle(
            raw_query=clean_query,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            keywords=keywords,
            rewritten_queries=rewritten_queries,
            filters={},
            search_plan={
                "use_dense": True,
                "retrieve_mode": "paper_lookup" if intent == "paper_lookup" else "knowledge_qa",
            },
            answer_plan={
                "style": "paper_list" if intent == "paper_lookup" else "direct_answer",
                "cite_evidence": True,
            },
        )

    def handle_query(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        try:
            query_bundle = self.parse_query(query)
            effective_top_k = top_k if isinstance(top_k, int) and top_k > 0 else self.config.top_k
            docs = self._retrieve_docs(query_bundle, effective_top_k)

            if not docs:
                return self._build_response(
                    success=True,
                    intent=query_bundle.intent,
                    answer="当前文献库中没有检索到与该问题直接相关的证据。",
                    answer_type="direct_answer" if query_bundle.intent != "paper_lookup" else "paper_list",
                    references=[],
                    snippets=[],
                    evidence_summary=[],
                    status="retrieval_failed",
                    confidence=0.0,
                    debug={
                        "intent_confidence": query_bundle.intent_confidence,
                        "rewritten_queries": query_bundle.rewritten_queries,
                        "section_weights": resolve_section_weights(query_bundle.intent),
                    },
                )

            references, chunks = self._format_docs(docs)
            response = self._dispatch_by_intent(query_bundle, chunks, references, effective_top_k)
            response["top_k_used"] = effective_top_k
            response["prompt_log_path"] = str(
                save_prompt_log(
                    self.config.work_dir,
                    query=query,
                    query_bundle=query_bundle,
                    chunks=chunks,
                    papers=response.get("papers", []),
                    answer=response["answer"],
                    status=response.get("status", "ok"),
                    debug=response.get("debug", {}),
                )
            )
            return response
        except Exception as exc:
            return self._build_response(False, "unknown", "", error_msg=str(exc))

    def _dispatch_by_intent(
        self,
        query_bundle: QueryBundle,
        chunks: list[dict[str, Any]],
        references: list[dict[str, Any]],
        top_k: int,
    ) -> Dict[str, Any]:
        if query_bundle.intent == "paper_lookup":
            papers = aggregate_to_papers(query_bundle, chunks, top_papers=top_k)
            payload = generate_answer(query_bundle, papers)
            return self._build_response(
                success=True,
                intent=query_bundle.intent,
                answer=payload.answer_text,
                answer_type=payload.answer_type,
                references=references,
                snippets=chunks[:top_k],
                evidence_summary=payload.evidence_summary,
                papers=[paper.to_dict() for paper in payload.papers],
                status=payload.status,
                confidence=payload.confidence,
                debug={
                    **payload.debug,
                    "rewritten_queries": query_bundle.rewritten_queries,
                    "intent_confidence": query_bundle.intent_confidence,
                },
            )
        return self._synthesize_direct_answer(query_bundle, chunks)

    def _retrieve_docs(self, query_bundle: QueryBundle, top_k: int) -> list[Any]:
        retrieve_k = max(top_k, 6 if query_bundle.intent != "paper_lookup" else top_k)
        seen_chunk_ids: set[str] = set()
        merged_docs: list[Any] = []
        for rewritten_query in query_bundle.rewritten_queries[:3]:
            docs = self.vectorstore.similarity_search_with_score(rewritten_query, k=retrieve_k)
            for doc, score in docs:
                metadata = dict(doc.metadata or {})
                metadata["score"] = float(score)
                doc.metadata = metadata
                chunk_id = str(metadata.get("chunk_id") or f"{metadata.get('source')}::{metadata.get('page')}::{len(merged_docs)}")
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                merged_docs.append(doc)
        return merged_docs

    def _extract_entities(self, query: str) -> list[str]:
        entities: list[str] = []
        for match in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", query):
            if match not in entities:
                entities.append(match)
        for match in re.findall(r"\b[A-Za-z]+RAG\b", query, flags=re.IGNORECASE):
            normalized = match.strip()
            if normalized not in entities:
                entities.append(normalized)
        return entities[:8]

    def _extract_keywords(self, query: str) -> list[str]:
        tokens = re.split(r"[\s,，。；;：:（）()\[\]、]+", query.lower())
        return [token for token in tokens if len(token) >= 2][:12]

    def _build_rewritten_queries(self, query: str) -> list[str]:
        llm_rewrites = self._rewrite_query_with_llm(query)
        if llm_rewrites:
            merged = [query]
            for item in llm_rewrites:
                normalized = " ".join(str(item).split())
                if normalized and normalized not in merged:
                    merged.append(normalized)
            return merged
        return self._fallback_rewritten_queries(query)

    def _rewrite_query_with_llm(self, query: str) -> list[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You rewrite academic search queries for literature retrieval. "
                    "Return strict JSON only in the format "
                    '{"rewritten_queries": ["...", "..."]}. '
                    "Keep at most 3 rewrites, preserve meaning, and expand abbreviations when helpful.",
                ),
                (
                    "human",
                    'Query: {query}\nReturn JSON only, e.g. {"rewritten_queries": ["...", "..."]}',
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({"query": query}).strip()
            parsed = json.loads(raw)
            rewrites = parsed.get("rewritten_queries", [])
            if isinstance(rewrites, list):
                return [str(item) for item in rewrites if str(item).strip()][:3]
        except Exception:
            return []
        return []

    def _fallback_rewritten_queries(self, query: str) -> list[str]:
        alias_map = {
            "ppr": "Personalized PageRank",
            "hipporag": "HippoRAG",
            "hipporag 2": "HippoRAG 2",
            "graphrag": "GraphRAG",
        }
        rewrites = [query]
        lowered = query.lower()
        for key, value in alias_map.items():
            if key in lowered and value not in rewrites:
                rewrites.append(re.sub(key, value, query, flags=re.IGNORECASE))
        return rewrites[:3]

    def _synthesize_direct_answer(self, query_bundle: QueryBundle, chunks: list[dict[str, Any]]) -> Dict[str, Any]:
        section_weights = resolve_section_weights(query_bundle.intent)
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: (
                section_weights.get(str(chunk.get("chunk_type") or "body"), 1.0) * float(chunk.get("score", 0.0)),
                section_weights.get(str(chunk.get("section") or "body"), 1.0),
            ),
            reverse=True,
        )
        evidence_chunks = ranked_chunks[: min(6, len(ranked_chunks))]
        context_parts: list[str] = []
        evidence_summary: list[str] = []
        for index, chunk in enumerate(evidence_chunks, start=1):
            title = chunk.get("paper_title") or chunk.get("title") or chunk.get("source") or f"片段 {index}"
            page = chunk.get("page") or chunk.get("page_start")
            page_text = f"第 {page} 页" if page else "页码未知"
            answer_context = str(chunk.get("answer_context") or chunk.get("text") or "")
            raw_text = str(chunk.get("raw_text") or chunk.get("text") or "")
            context_parts.append(f"[证据 {index}] {title} | {page_text}\n{answer_context}")
            snippet = raw_text.replace("\n", " ").strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            evidence_summary.append(f"{title}：{page_text}，{snippet}")

        context = "\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个严谨的个人文献知识库问答助手。"
                    "请根据提供的文献证据直接回答问题，优先给出明确结论。"
                    "如果证据不足，请明确说明不确定，不要编造。"
                    "回答保持简洁，并指出核心依据。",
                ),
                ("human", "问题：{query}\n\n证据：\n{context}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({"query": query_bundle.raw_query, "context": context}).strip()
            if not answer:
                raise ValueError("empty answer")
            return self._build_response(
                success=True,
                intent=query_bundle.intent,
                answer=answer,
                answer_type="direct_answer",
                references=[],
                snippets=evidence_chunks,
                evidence_summary=evidence_summary,
                status="ok",
                confidence=min(0.92, 0.55 + 0.06 * len(evidence_chunks)),
                debug={
                    "rewritten_queries": query_bundle.rewritten_queries,
                    "intent_confidence": query_bundle.intent_confidence,
                    "retrieved_chunk_count": len(chunks),
                    "used_evidence_count": len(evidence_chunks),
                    "section_weights": section_weights,
                },
            )
        except Exception:
            return self._build_response(
                success=True,
                intent=query_bundle.intent,
                answer=self._fallback_direct_answer(query_bundle, evidence_chunks),
                answer_type="direct_answer",
                references=[],
                snippets=evidence_chunks,
                evidence_summary=evidence_summary,
                status="llm_fallback",
                confidence=0.45 if evidence_chunks else 0.0,
                debug={
                    "rewritten_queries": query_bundle.rewritten_queries,
                    "intent_confidence": query_bundle.intent_confidence,
                    "retrieved_chunk_count": len(chunks),
                    "used_evidence_count": len(evidence_chunks),
                    "section_weights": section_weights,
                },
            )

    def _fallback_direct_answer(self, query_bundle: QueryBundle, evidence_chunks: list[dict[str, Any]]) -> str:
        if not evidence_chunks:
            return f"当前文献库中暂时没有找到足以回答“{query_bundle.raw_query}”的直接证据。"
        top = evidence_chunks[0]
        title = top.get("paper_title") or top.get("title") or top.get("source") or "未知文献"
        page = top.get("page") or top.get("page_start")
        page_text = f"第 {page} 页" if page else "页码未知"
        snippet = str(top.get("raw_text") or top.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        return (
            f"我已经在文献库中找到了与“{query_bundle.raw_query}”最相关的证据。"
            f"当前最强证据来自《{title}》({page_text})。\n\n依据：{snippet}"
        )

    def _format_docs(self, docs: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        references: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for rank, doc in enumerate(docs, start=1):
            metadata = dict(doc.metadata or {})
            source_path = str(metadata.get("source_path") or metadata.get("source") or "unknown")
            title = str(metadata.get("paper_title") or metadata.get("title") or Path(source_path).name)
            display_text = str(doc.page_content or "")
            raw_text = str(metadata.get("raw_text") or display_text)
            answer_context = str(metadata.get("context_window") or raw_text)
            item = {
                "rank": rank,
                "chunk_id": metadata.get("chunk_id"),
                "paper_id": metadata.get("paper_id"),
                "paper_title": title,
                "title": title,
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "source": source_path,
                "source_path": source_path,
                "rel_path": metadata.get("rel_path"),
                "page": metadata.get("page_1based") or metadata.get("page") or metadata.get("page_start"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "section": metadata.get("section"),
                "chunk_type": metadata.get("chunk_type"),
                "score": float(metadata.get("score", 0.0)),
                "content": raw_text,
                "text": raw_text,
                "raw_text": raw_text,
                "answer_context": answer_context,
                "context_before": metadata.get("context_before", ""),
                "context_after": metadata.get("context_after", ""),
                "chunk_local_index": metadata.get("chunk_local_index"),
                "chunk_local_total": metadata.get("chunk_local_total"),
            }
            chunks.append(item)
            references.append(
                {
                    "title": title,
                    "source_path": source_path,
                    "page": item["page"],
                    "content": raw_text,
                    "score": item["score"],
                }
            )
        return references, chunks

    def _build_response(
        self,
        success: bool,
        intent: str,
        answer: str,
        *,
        answer_type: str = "paper_list",
        references: list[dict[str, Any]] | None = None,
        snippets: list[dict[str, Any]] | None = None,
        evidence_summary: list[str] | None = None,
        papers: list[dict[str, Any]] | None = None,
        status: str = "ok",
        confidence: float = 0.0,
        debug: dict[str, Any] | None = None,
        error_msg: str | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "success": success,
            "intent": intent,
            "answer": answer,
            "answer_type": answer_type,
            "references": references or [],
            "snippets": snippets or [],
            "papers": papers or [],
            "evidence_summary": evidence_summary or [],
            "status": status,
            "confidence": confidence,
            "debug": debug or {},
        }
        if not success:
            payload["error"] = {"message": error_msg or "Unknown error"}
        return payload
