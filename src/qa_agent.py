"""Current Zotero QA agent with intent routing and LLM-assisted query rewrite."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.aggregator import aggregate_to_papers, generate_answer, resolve_section_weights
from src.config import AppConfig
from src.domain_models import QueryBundle
from src.indexer import build_llm, get_vectorstore
from src.manifest import prepare_manifest_snapshot
from src.metadata_store import MetadataStore
from src.prompt_logger import save_prompt_log


def _serialize_messages(messages: list[Any]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in messages:
        serialized.append(
            {
                "type": getattr(message, "type", message.__class__.__name__),
                "content": str(getattr(message, "content", "")),
            }
        )
    return serialized


def _to_relevance_score(raw_score: Any) -> float:
    """
    Normalize vectorstore scores to a "higher is better" relevance score.
    FAISS commonly returns distances, so we convert them into a bounded value.
    """
    try:
        numeric_score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    if numeric_score < 0:
        return numeric_score
    return 1.0 / (1.0 + numeric_score)


def _uses_paper_list_answer(intent: str) -> bool:
    return intent in {"paper_lookup", "survey"}


VALID_INTENTS = {"fact_qa", "comparison", "definition", "survey", "paper_lookup"}


def _normalize_filter_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，;\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                normalized.append(text)
        return normalized
    return []


def _normalize_year_filter(raw_filters: dict[str, Any]) -> dict[str, int] | None:
    date_range = raw_filters.get("date_range")
    if not isinstance(date_range, dict):
        return None
    normalized: dict[str, int] = {}
    for key in ("from", "to"):
        value = date_range.get(key)
        if value in (None, ""):
            continue
        try:
            normalized[key] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized or None


def _normalize_filters(raw_filters: dict[str, Any] | None) -> dict[str, Any]:
    base = raw_filters if isinstance(raw_filters, dict) else {}
    normalized: dict[str, Any] = {}
    collections = _normalize_filter_text_list(base.get("collections"))
    tags = _normalize_filter_text_list(base.get("tags"))
    authors = _normalize_filter_text_list(base.get("authors"))
    if collections:
        normalized["collections"] = collections
    if tags:
        normalized["tags"] = tags
    if authors:
        normalized["authors"] = authors
    year_range = _normalize_year_filter(base)
    if year_range:
        normalized["date_range"] = year_range
    return normalized


def _contains_any_text(candidates: Any, expected_values: list[str]) -> bool:
    if not expected_values:
        return True
    if isinstance(candidates, str):
        candidate_list = [candidates]
    elif isinstance(candidates, (list, tuple, set)):
        candidate_list = [str(item or "") for item in candidates]
    else:
        candidate_list = []
    lowered_candidates = [item.strip().lower() for item in candidate_list if str(item or "").strip()]
    lowered_expected = [item.strip().lower() for item in expected_values if item.strip()]
    if not lowered_candidates:
        return False
    return any(expected in candidate for expected in lowered_expected for candidate in lowered_candidates)


def _normalize_text_values(values: Any) -> list[str]:
    if isinstance(values, str):
        items = [values]
    elif isinstance(values, (list, tuple, set)):
        items = [str(item or "") for item in values]
    else:
        items = []
    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


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
        self.metadata_store = MetadataStore(self.config.metadata_db_path)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.top_k})
        self.llm = build_llm(self.config)
        self._last_llm_debug: dict[str, Any] = {}

    def parse_query(self, query: str, filters: dict[str, Any] | None = None) -> QueryBundle:
        clean_query = " ".join(query.split())
        entities = self._extract_entities(clean_query)
        keywords = self._extract_keywords(clean_query)
        self._last_llm_debug = {"query": clean_query}
        normalized_filters = _normalize_filters(filters)

        intent, confidence = self._classify_intent(clean_query)

        rewritten_queries = self._build_rewritten_queries(clean_query)
        return QueryBundle(
            raw_query=clean_query,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            keywords=keywords,
            rewritten_queries=rewritten_queries,
            filters=normalized_filters,
            search_plan={
                "use_dense": True,
                "retrieve_mode": "paper_lookup" if intent == "paper_lookup" else "knowledge_qa",
            },
            answer_plan={
                "style": "paper_list" if _uses_paper_list_answer(intent) else "direct_answer",
                "cite_evidence": True,
            },
        )

    def _classify_intent(self, query: str) -> tuple[str, float]:
        llm_intent, llm_confidence = self._classify_intent_with_llm(query)
        if llm_intent in VALID_INTENTS:
            self._last_llm_debug["intent_resolution"] = {
                "source": "llm",
                "intent": llm_intent,
                "confidence": llm_confidence,
            }
            return llm_intent, llm_confidence
        rule_intent, rule_confidence = self._classify_intent_with_rules(query)
        self._last_llm_debug["intent_resolution"] = {
            "source": "rules_fallback",
            "intent": rule_intent,
            "confidence": rule_confidence,
        }
        return rule_intent, rule_confidence

    def _classify_intent_with_llm(self, query: str) -> tuple[str | None, float]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You classify user queries for a personal academic literature assistant. "
                    "Choose exactly one intent from: fact_qa, comparison, definition, survey, paper_lookup. "
                    "Academic papers follow structured sections such as title, abstract, introduction, method, experiment/results, conclusion, and references. "
                    "When a query asks which papers/articles use, adopt, propose, contain, or mention a method/topic in a paper section such as method, approach, experiment, or abstract, classify it as paper_lookup. "
                    "Use paper_lookup for queries asking which papers/articles/works use, discuss, propose, compare, contain, or mention a method/topic. "
                    "Use survey for broad overview or review requests. "
                    "Return strict JSON only in the format "
                    '{{"intent":"paper_lookup","confidence":0.92}}.',
                ),
                (
                    "human",
                    "Query: {query}\nReturn JSON only.",
                ),
            ]
        )
        messages = prompt.format_messages(query=query)
        self._last_llm_debug["intent_classification"] = {
            "prompt_messages": _serialize_messages(messages),
        }
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({"query": query}).strip()
            self._last_llm_debug["intent_classification"]["raw_response"] = raw
            parsed = json.loads(raw)
            intent = str(parsed.get("intent", "")).strip()
            confidence = float(parsed.get("confidence", 0.0))
            if intent in VALID_INTENTS:
                self._last_llm_debug["intent_classification"]["parsed"] = {
                    "intent": intent,
                    "confidence": confidence,
                    "valid": True,
                }
                return intent, max(0.0, min(confidence, 1.0))
            self._last_llm_debug["intent_classification"]["parsed"] = {
                "intent": intent,
                "confidence": confidence,
                "valid": False,
            }
        except Exception as exc:
            self._last_llm_debug["intent_classification"]["error"] = str(exc)
            return None, 0.0
        return None, 0.0

    def _classify_intent_with_rules(self, query: str) -> tuple[str, float]:
        lowered = query.lower()
        intent = "fact_qa"
        confidence = 0.55
        for candidate, patterns in INTENT_PATTERNS:
            if any(pattern.lower() in lowered for pattern in patterns):
                intent = candidate
                confidence = 0.85
                break
        if query.startswith(FACT_PREFIXES) or any(hint in lowered for hint in FACT_HINTS):
            intent = "fact_qa"
            confidence = max(confidence, 0.8)
        self._last_llm_debug["intent_rules"] = {
            "intent": intent,
            "confidence": confidence,
        }
        return intent, confidence

    def handle_query(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            query_bundle = self.parse_query(query, filters=filters)
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
                        "hard_filters": query_bundle.filters,
                        "section_weights": resolve_section_weights(query_bundle.intent),
                        "llm": self._last_llm_debug,
                    },
                )

            references, chunks = self._format_docs(docs)
            response = self._dispatch_by_intent(query_bundle, chunks, references, effective_top_k)
            response.setdefault("debug", {})
            response["debug"]["llm"] = self._last_llm_debug
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
        if _uses_paper_list_answer(query_bundle.intent):
            paper_limit = max(top_k, 8)
            papers = aggregate_to_papers(query_bundle, chunks, top_papers=paper_limit)
            payload = generate_answer(query_bundle, papers, self.llm)
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
                    "hard_filters": query_bundle.filters,
                },
            )
        return self._synthesize_direct_answer(query_bundle, chunks)

    def _retrieve_docs(self, query_bundle: QueryBundle, top_k: int) -> list[Any]:
        total_index_docs = self._estimate_vectorstore_doc_count()
        candidate_scope = self.metadata_store.resolve_candidate_scope(query_bundle.filters)
        candidate_row_ids = candidate_scope["faiss_row_ids"] if query_bundle.filters else None
        if _uses_paper_list_answer(query_bundle.intent):
            retrieve_k = max(top_k * 4, 12)
        else:
            retrieve_k = max(top_k, 6)
        if query_bundle.filters:
            retrieve_k = min(max(retrieve_k, 12), max(1, candidate_scope["chunk_count"]))
        seen_chunk_ids: set[str] = set()
        merged_docs: list[Any] = []
        diagnostics: dict[str, Any] = {
            "hard_filters": query_bundle.filters,
            "candidate_source": candidate_scope["candidate_source"],
            "candidate_paper_count": candidate_scope["paper_count"],
            "candidate_chunk_count": candidate_scope["chunk_count"],
            "retrieve_k": retrieve_k,
            "total_index_docs": total_index_docs,
            "rewritten_queries": query_bundle.rewritten_queries[:3],
            "per_query": [],
            "total_raw_hits": 0,
            "total_kept_hits": 0,
            "rejection_counts": {
                "deduplicated": 0,
            },
        }
        if query_bundle.filters and not candidate_row_ids:
            diagnostics["final_ranked_hits"] = 0
            self._last_llm_debug["hard_filter_diagnostics"] = diagnostics
            self._log_hard_filter_diagnostics(query_bundle.raw_query, diagnostics)
            return []
        for rewritten_query in query_bundle.rewritten_queries[:3]:
            if candidate_row_ids is None:
                docs = self.vectorstore.similarity_search_with_score(rewritten_query, k=retrieve_k)
                search_scope = "fullindex"
            else:
                docs = self._similarity_search_with_score_in_subset(rewritten_query, candidate_row_ids, retrieve_k)
                search_scope = "subindex"
            diagnostics["total_raw_hits"] += len(docs)
            scored_docs: list[tuple[Any, float]] = []
            for doc, score in docs:
                metadata = dict(doc.metadata or {})
                metadata["raw_score"] = float(score)
                metadata["score"] = _to_relevance_score(score)
                doc.metadata = metadata
                scored_docs.append((doc, float(metadata["score"])))
            per_query_stats = {
                "query": rewritten_query,
                "search_scope": search_scope,
                "raw_hits": len(docs),
                "kept_hits": 0,
                "rejected": {
                    "deduplicated": 0,
                },
            }
            scored_docs.sort(key=lambda item: item[1], reverse=True)
            for doc, _ in scored_docs:
                metadata = dict(doc.metadata or {})
                chunk_id = str(metadata.get("chunk_id") or "").strip()
                if not chunk_id:
                    chunk_id = f"{metadata.get('source')}::{metadata.get('page')}::{len(merged_docs)}"
                if chunk_id in seen_chunk_ids:
                    diagnostics["rejection_counts"]["deduplicated"] += 1
                    per_query_stats["rejected"]["deduplicated"] += 1
                    continue
                seen_chunk_ids.add(chunk_id)
                merged_docs.append(doc)
                diagnostics["total_kept_hits"] += 1
                per_query_stats["kept_hits"] += 1
            diagnostics["per_query"].append(per_query_stats)
        merged_docs.sort(
            key=lambda doc: float((doc.metadata or {}).get("score", 0.0)),
            reverse=True,
        )
        diagnostics["final_ranked_hits"] = len(merged_docs)
        self._last_llm_debug["hard_filter_diagnostics"] = diagnostics
        self._log_hard_filter_diagnostics(query_bundle.raw_query, diagnostics)
        return merged_docs

    def _estimate_vectorstore_doc_count(self) -> int:
        mapping = getattr(self.vectorstore, "index_to_docstore_id", None)
        if isinstance(mapping, dict):
            return len(mapping)
        if isinstance(mapping, list):
            return len(mapping)
        return 0

    def _similarity_search_with_score_in_subset(
        self,
        query: str,
        candidate_row_ids: list[int],
        k: int,
    ) -> list[tuple[Any, float]]:
        if not candidate_row_ids:
            return []

        query_vector = np.asarray(self._embed_query(query), dtype="float32")
        row_ids = [int(row_id) for row_id in candidate_row_ids]
        sub_vectors: list[np.ndarray] = []
        valid_row_ids: list[int] = []
        for row_id in row_ids:
            try:
                vector = np.asarray(self.vectorstore.index.reconstruct(int(row_id)), dtype="float32")
            except Exception:
                continue
            if vector.ndim != 1:
                vector = vector.reshape(-1)
            sub_vectors.append(vector)
            valid_row_ids.append(int(row_id))

        if not valid_row_ids:
            return []

        matrix = np.vstack(sub_vectors).astype("float32")
        dim = int(matrix.shape[1])
        metric_type = getattr(self.vectorstore.index, "metric_type", faiss.METRIC_L2)
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            sub_index = faiss.IndexFlatIP(dim)
        else:
            sub_index = faiss.IndexFlatL2(dim)
        sub_index.add(matrix)

        limit = min(max(int(k), 1), len(valid_row_ids))
        distances, indices = sub_index.search(query_vector.reshape(1, -1), limit)

        results: list[tuple[Any, float]] = []
        for local_idx, distance in zip(indices[0], distances[0]):
            if int(local_idx) < 0:
                continue
            row_id = valid_row_ids[int(local_idx)]
            doc = self._get_doc_by_row_id(row_id)
            if doc is None:
                continue
            results.append((doc, float(distance)))
        return results

    def _embed_query(self, query: str) -> list[float]:
        embedding_function = getattr(self.vectorstore, "embedding_function", None)
        if embedding_function is not None:
            if hasattr(embedding_function, "embed_query"):
                return list(embedding_function.embed_query(query))
            if callable(embedding_function):
                return list(embedding_function(query))
        embeddings = getattr(self.vectorstore, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "embed_query"):
            return list(embeddings.embed_query(query))
        raise RuntimeError("Vectorstore embedding function is unavailable.")

    def _get_doc_by_row_id(self, row_id: int) -> Any | None:
        mapping = getattr(self.vectorstore, "index_to_docstore_id", None)
        docstore = getattr(self.vectorstore, "docstore", None)
        if mapping is None or docstore is None:
            return None

        if isinstance(mapping, dict):
            docstore_id = mapping.get(int(row_id))
        elif isinstance(mapping, list):
            if int(row_id) < 0 or int(row_id) >= len(mapping):
                return None
            docstore_id = mapping[int(row_id)]
        else:
            return None
        if docstore_id is None:
            return None
        internal_dict = getattr(docstore, "_dict", None)
        if isinstance(internal_dict, dict):
            return internal_dict.get(docstore_id)
        search_fn = getattr(docstore, "search", None)
        if callable(search_fn):
            try:
                return search_fn(docstore_id)
            except Exception:
                return None
        return None

    def _log_hard_filter_diagnostics(self, raw_query: str, diagnostics: dict[str, Any]) -> None:
        hard_filters = diagnostics.get("hard_filters") or {}
        if hard_filters:
            print(
                "[HardFilter] "
                f"query={raw_query!r} "
                f"filters={hard_filters} "
                f"candidate_source={diagnostics.get('candidate_source')} "
                f"candidate_papers={diagnostics.get('candidate_paper_count', 0)} "
                f"candidate_chunks={diagnostics.get('candidate_chunk_count', 0)} "
                f"total_index_docs={diagnostics.get('total_index_docs', 0)} "
                f"raw_hits={diagnostics.get('total_raw_hits', 0)} "
                f"kept_hits={diagnostics.get('total_kept_hits', 0)} "
                f"final_hits={diagnostics.get('final_ranked_hits', 0)}"
            )
            print(
                "[HardFilter] "
                f"rejections={diagnostics.get('rejection_counts', {})}"
            )
        else:
            print(
                "[HardFilter] "
                f"query={raw_query!r} filters=<none> "
                f"candidate_source={diagnostics.get('candidate_source')} "
                f"total_index_docs={diagnostics.get('total_index_docs', 0)} "
                f"raw_hits={diagnostics.get('total_raw_hits', 0)} "
                f"final_hits={diagnostics.get('final_ranked_hits', 0)}"
            )

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
        merged = [query]
        llm_rewrites = self._rewrite_query_with_llm(query)
        for item in llm_rewrites:
            normalized = " ".join(str(item).split())
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    def _rewrite_query_with_llm(self, query: str) -> list[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You rewrite academic search queries for literature retrieval. "
                    "Assume papers follow common academic structure: title, abstract, introduction, method, experiment/results, conclusion, references. "
                    "When the user asks about papers that use a method/topic, rewrite the query so it targets method/approach usage in paper body sections rather than papers whose main topic is that method itself. "
                    "Preserve section constraints such as method, abstract, experiment when they are present in the query. "
                    "Prefer terminology that appears in academic prose, such as 'in the method section', 'uses', 'adopts', 'based on', 'applies'. "
                    "Return strict JSON only in the format "
                    '{{"rewritten_queries": ["...", "..."]}}. '
                    "Keep at most 3 rewrites, preserve meaning, and expand abbreviations when helpful.",
                ),
                (
                    "human",
                    'Query: {query}\nReturn JSON only, e.g. {{"rewritten_queries": ["...", "..."]}}',
                ),
            ]
        )
        messages = prompt.format_messages(query=query)
        self._last_llm_debug["query_rewrite"] = {
            "prompt_messages": _serialize_messages(messages),
        }
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({"query": query}).strip()
            self._last_llm_debug["query_rewrite"]["raw_response"] = raw
            parsed = json.loads(raw)
            rewrites = parsed.get("rewritten_queries", [])
            if isinstance(rewrites, list):
                cleaned = [str(item) for item in rewrites if str(item).strip()][:3]
                self._last_llm_debug["query_rewrite"]["parsed"] = {
                    "rewritten_queries": cleaned,
                    "valid": True,
                }
                return cleaned
            self._last_llm_debug["query_rewrite"]["parsed"] = {
                "rewritten_queries": [],
                "valid": False,
            }
        except Exception as exc:
            self._last_llm_debug["query_rewrite"]["error"] = str(exc)
            return []
        return []

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
            context_parts.append(f"[{index}] {title} | {page_text}\n{answer_context}")
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
                    "回答保持简洁。"
                    "如需引用证据，只能使用编号引用 [1]、[2] 这种形式。"
                    "不要使用“证据1”“资料1”“来源1”等写法。"
                    "不要输出参考文献列表或重复展开文献标题、作者、年份。",
                ),
                ("human", "问题：{query}\n\n证据：\n{context}\n\n请直接输出回答正文。"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({"query": query_bundle.raw_query, "context": context}).strip()
            if not answer:
                raise ValueError("empty answer")
            answer = self._sanitize_direct_answer(answer, len(evidence_chunks))
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
                    "hard_filters": query_bundle.filters,
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
                    "hard_filters": query_bundle.filters,
                },
            )

    def _sanitize_direct_answer(self, answer: str, evidence_count: int) -> str:
        cleaned = str(answer or "").strip()
        if not cleaned:
            return ""

        cleaned = re.sub(r"\[\s*证据\s*(\d+)\s*\]", r"[\1]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"（\s*证据\s*(\d+)\s*）", r"[\1]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(\s*证据\s*(\d+)\s*\)", r"[\1]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"证据\s*(\d+)", r"[\1]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"来源\s*(\d+)", r"[\1]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\[\s*(\d+)\s*\]",
            lambda m: f"[{m.group(1)}]" if 1 <= int(m.group(1)) <= evidence_count else "",
            cleaned,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if evidence_count > 0 and not re.search(r"\[(\d+)\]", cleaned):
            cleaned = f"{cleaned} [1]".strip()
        return cleaned

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
            f"当前结论主要基于 [1]。\n\n依据：{snippet}"
        )

    def _ensure_numbered_citations(self, answer: str, source_count: int) -> str:
        cleaned = str(answer or "").strip()
        if source_count <= 0:
            return cleaned
        if re.search(r"\[(\d+)\]", cleaned):
            return cleaned
        suffix = "".join(f"[{index}]" for index in range(1, min(source_count, 3) + 1))
        return f"{cleaned} {suffix}".strip()

    def _resolve_attachment_key(self, metadata: dict[str, Any]) -> str | None:
        attachment_key = str(metadata.get("attachment_key") or "").strip()
        if attachment_key:
            return attachment_key

        source_path = str(metadata.get("source_path") or metadata.get("source") or "").strip()
        if "storage" not in source_path:
            return None

        try:
            parent_name = Path(source_path).parent.name
        except Exception:
            return None
        return parent_name or None

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
                "tags": metadata.get("tags"),
                "collections": metadata.get("collections"),
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
                "raw_score": metadata.get("raw_score"),
                "content": raw_text,
                "text": raw_text,
                "raw_text": raw_text,
                "answer_context": answer_context,
                "context_before": metadata.get("context_before", ""),
                "context_after": metadata.get("context_after", ""),
                "chunk_local_index": metadata.get("chunk_local_index"),
                "chunk_local_total": metadata.get("chunk_local_total"),
                "attachment_key": self._resolve_attachment_key(metadata),
            }
            chunks.append(item)
            references.append(
                {
                    "title": title,
                    "source_path": source_path,
                    "page": item["page"],
                    "content": raw_text,
                    "score": item["score"],
                    "raw_score": item["raw_score"],
                    "attachment_key": self._resolve_attachment_key(metadata),
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
        effective_references = references or []
        effective_snippets = snippets or []
        effective_evidence_summary = evidence_summary or []
        effective_source_count = len(effective_references) if effective_references else len(effective_snippets)
        normalized_answer = self._ensure_numbered_citations(answer, effective_source_count) if success else answer
        payload: Dict[str, Any] = {
            "success": success,
            "intent": intent,
            "answer": normalized_answer,
            "answer_type": answer_type,
            "references": effective_references,
            "snippets": effective_snippets,
            "papers": papers or [],
            "evidence_summary": effective_evidence_summary,
            "status": status,
            "confidence": confidence,
            "debug": debug or {},
        }
        if not success:
            payload["error"] = {"message": error_msg or "Unknown error"}
        return payload
