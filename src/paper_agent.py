from pathlib import Path
from typing import Any, Dict

from src.aggregator import aggregate_to_papers
from src.answerer import generate_answer
from src.bootstrap import prepare_manifest_snapshot
from src.config import AppConfig
from src.indexer import get_vectorstore
from src.schema import QueryBundle


class ZoteroAgent:
    def __init__(self, config: AppConfig):
        print("Initializing Zotero Agent...")
        self.config = config

        prepare_manifest_snapshot(self.config)
        self.vectorstore = get_vectorstore(self.config)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.top_k})

    def parse_query(self, query: str) -> QueryBundle:
        return QueryBundle(
            raw_query=query,
            intent="paper_search",
            rewritten_queries=[],
            filters={},
            search_plan={"use_dense": True},
        )

    def handle_query(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        try:
            query_bundle = self.parse_query(query)
            effective_top_k = top_k if isinstance(top_k, int) and top_k > 0 else self.config.top_k
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": effective_top_k})
            docs = retriever.invoke(query)

            if not docs:
                return self._build_response(
                    success=True,
                    intent=query_bundle.intent,
                    answer="在当前文献库中未检索到与问题相关的片段。",
                    references=[],
                    snippets=[],
                )

            references, chunks = self._format_docs(docs)
            papers = aggregate_to_papers(query_bundle, chunks, top_papers=effective_top_k)
            answer_payload = generate_answer(query_bundle, papers)

            response = self._build_response(
                success=True,
                intent=query_bundle.intent,
                answer=answer_payload.answer_text,
                references=references,
                snippets=chunks,
            )
            response["papers"] = [paper.to_dict() for paper in answer_payload.papers]
            response["confidence"] = answer_payload.confidence
            response["status"] = answer_payload.status
            response["debug"] = answer_payload.debug
            response["top_k_used"] = effective_top_k
            return response
        except Exception as e:
            return self._build_response(False, "unknown", "", error_msg=str(e))

    def _format_docs(self, docs) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        references: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []

        for idx, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            source_path = str(metadata.get("source_path") or metadata.get("source") or "unknown")
            file_name = metadata.get("file_name") or (Path(source_path).name if source_path != "unknown" else "unknown")
            paper_title = str(metadata.get("paper_title") or Path(file_name).stem)

            page = metadata.get("page_1based")
            if not isinstance(page, int):
                page_raw = metadata.get("page")
                page = page_raw + 1 if isinstance(page_raw, int) else None

            chunk_id = str(metadata.get("chunk_id") or f"{file_name}#chunk-{idx}")
            chunk_rank = metadata.get("chunk_rank")
            if not isinstance(chunk_rank, int):
                chunk_rank = idx

            ref_info = {"source": file_name, "source_path": source_path, "page": page, "paper_title": paper_title}
            if ref_info not in references:
                references.append(ref_info)

            chunks.append(
                {
                    "id": idx,
                    "chunk_id": chunk_id,
                    "rank": chunk_rank,
                    "score": 1.0 / (idx + 1),
                    "paper_id": metadata.get("paper_id"),
                    "paper_title": paper_title,
                    "authors": metadata.get("authors"),
                    "year": metadata.get("year"),
                    "venue": metadata.get("venue"),
                    "section": metadata.get("section"),
                    "chunk_type": metadata.get("chunk_type"),
                    "page_start": metadata.get("page_start", page),
                    "page_end": metadata.get("page_end", page),
                    "title": paper_title,
                    "source": file_name,
                    "source_path": source_path,
                    "rel_path": metadata.get("rel_path"),
                    "page": page,
                    "content": doc.page_content or "",
                    "text": doc.page_content or "",
                }
            )

        return references, chunks

    def _build_response(
        self,
        success: bool,
        intent: str,
        answer: str,
        references: list | None = None,
        snippets: list | None = None,
        error_msg: str | None = None,
    ) -> Dict[str, Any]:
        result = {
            "success": success,
            "intent": intent,
            "answer": answer,
            "references": references or [],
            "snippets": snippets or [],
        }
        if not success:
            result["error"] = {"message": error_msg}
        return result
