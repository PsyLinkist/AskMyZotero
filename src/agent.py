from pathlib import Path
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入已有的底层模块
from src.config import AppConfig
from src.indexer import get_vectorstore, build_llm
from src.bootstrap import prepare_manifest_snapshot
from src.document_view import group_snippets_by_paper


class ZoteroAgent:
    """
    统一的 Agent 接口层。
    负责接收前端/外部请求，解析意图，调用 RAG 链路，并返回结构化结果。
    """
    def __init__(self, config: AppConfig):
        """
        直接接收组装好的 AppConfig 数据结构，不再处理任何参数解析逻辑。
        """
        print("🤖 正在初始化 Zotero Agent...")
        self.config = config

        # 先做启动期快照更新，确保后续索引基于最新文献目录
        prepare_manifest_snapshot(self.config)

        # 初始化核心组件：向量库与 LLM
        self.vectorstore = get_vectorstore(self.config)
        self.llm = build_llm(self.config)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.top_k})

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        [任务2] 查询解析函数
        当前为基础版本，后续可接入 LLM 判断是否需要 RAG、提取关键词等。
        """
        return {
            "raw_query": query,
            "intent": "paper_search",
            "keywords": [],
            "need_rag": True
        }

    def handle_query(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        """
        [任务4] 核心交互入口，供前端或外部直接调用。
        返回标准化的 JSON/字典 结构。
        """
        try:
            # 1. 解析查询意图
            parsed_info = self.parse_query(query)
            
            # 2. 检索相关文献片段（优先使用请求传入的 top_k）
            effective_top_k = top_k if isinstance(top_k, int) and top_k > 0 else self.config.top_k
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": effective_top_k})
            docs = retriever.invoke(query)
            
            if not docs:
                return self._build_response(
                    success=True, 
                    intent=parsed_info["intent"], 
                    answer="在当前文献库中未能检索到与您问题相关的片段。",
                    references=[],
                    snippets=[]
                )

            # 3. 组装上下文并格式化参考信息
            formatted_context, references, snippets = self._format_docs_for_agent(docs)
            papers = group_snippets_by_paper(snippets)

            # 4. 构建并调用 LLM 生成回答
            system_prompt = (
                "你是一个严谨的学术科研助手。"
                "请仅根据给定文献片段作答，不要编造。"
                "若检索结果不为空，必须先列出最相关的论文名称（可多篇）并简述其与问题的关系，再给出结论。"
                "只有在检索片段完全无法支持任何结论时，才回答“根据当前文献库无法回答该问题”。"
                "\n\n检索到的文献片段如下：\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": formatted_context, "input": query})

            # 5. 返回结构化结果
            response = self._build_response(
                success=True,
                intent=parsed_info["intent"],
                answer=answer,
                references=references,
                snippets=snippets
            )
            response["papers"] = papers
            response["top_k_used"] = effective_top_k
            return response

        except Exception as e:
            # 捕获异常并返回友好的错误结构
            return self._build_response(
                success=False,
                intent="unknown",
                answer="",
                error_msg=str(e)
            )

    def _format_docs_for_agent(self, docs) -> tuple[str, list, list]:
        """
        内部辅助函数：将 LangChain 的 Document 对象转为大模型上下文和前端展示所需的结构。
        """
        formatted_parts = []
        references = []
        snippets = []

        for idx, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}

            source_path = str(metadata.get("source_path") or metadata.get("source") or "unknown")
            file_name = metadata.get("file_name") or (Path(source_path).name if source_path != "unknown" else "unknown")

            page = metadata.get("page_1based")
            if not isinstance(page, int):
                page_raw = metadata.get("page")
                page = page_raw + 1 if isinstance(page_raw, int) else None

            chunk_id = str(metadata.get("chunk_id") or f"{file_name}#chunk-{idx}")
            chunk_rank = metadata.get("chunk_rank")
            if not isinstance(chunk_rank, int):
                chunk_rank = idx
            
            header = f"[片段 {idx}] source={file_name}"
            if page is not None:
                header += f", page={page}"
            formatted_parts.append(f"{header}\n{doc.page_content}")
            
            ref_info = {"source": file_name, "source_path": source_path, "page": page}
            if ref_info not in references:
                references.append(ref_info)
                
            snippets.append({
                "id": idx,
                "chunk_id": chunk_id,
                "rank": chunk_rank,
                "title": file_name,
                "source": file_name,
                "source_path": source_path,
                "page": page,
                "content": doc.page_content or "",
            })

        return "\n\n".join(formatted_parts), references, snippets

    def _build_response(self, success: bool, intent: str, answer: str, references: list = None, snippets: list = None, error_msg: str = None) -> Dict[str, Any]:
        """统一封装返回格式"""
        result = {
            "success": success,
            "intent": intent,
            "answer": answer,
            "references": references or [],
            "snippets": snippets or []
        }
        if not success:
            result["error"] = {"message": error_msg}
        return result