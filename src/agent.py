from typing import List, Dict, Any
from src.schemas import QueryRequest, QueryResponse, ReferenceSnippet
from src.indexer import api_ask_once  # 调用底层检索逻辑

def analyze_intent(query: str) -> Dict[str, Any]:
    """
    [占位函数] 任务 2 的核心：意图识别
    未来 Agent 同学将在这里调用 LLM，分析 query 中是否包含年份、会议等过滤条件。
    目前先返回一个基础的意图。
    """
    # 假设未来这里会解析出 filters = {"year": {">=": 2023}, "venue": "顶会"}
    return {
        "intent_type": "paper_search",
        "filters": {} 
    }

def format_references(raw_docs: List[Any]) -> List[ReferenceSnippet]:
    """
    [核心清洗函数] 把底层的脏数据变成前端能用的漂亮卡片数据。
    """
    snippets = []
    for i, doc in enumerate(raw_docs):
        # 兼容可能返回的不同格式（字典或 LangChain 的 Document 对象）
        metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
        page_content = doc.page_content if hasattr(doc, 'page_content') else doc.get('page_content', '')
        
        # 强行组装成 schemas.py 要求的 ReferenceSnippet 格式
        snippet = ReferenceSnippet(
            title=metadata.get("title", f"未知文献 {i+1}"),
            authors=metadata.get("authors", "未知作者"),
            venue=metadata.get("venue", "Zotero 库"),
            year=metadata.get("year", None),
            abstract=page_content[:200] + "...", # 截取前200字作为摘要显示
            source_path=metadata.get("source", "未知路径"),
            page=metadata.get("page", None),
            score=metadata.get("score", 0.0)
        )
        snippets.append(snippet)
    return snippets

def run_agent(request: QueryRequest, rag_chain: Any) -> QueryResponse:
    """
    [主入口] Server 会把前端的请求直接塞给这个函数。
    """
    print(f"🤖 [Agent] 接管请求，开始处理...")
    
    try:
        # 1. 意图分析 (目前是 Mock)
        intent_info = analyze_intent(request.query)
        
        # 2. 调用 Indexer 执行检索和问答
        # 注意：这里需要确认indexer.py中的 api_ask_once 的返回值里是否包含了原始文档(source_documents)
        result = api_ask_once(rag_chain, request.query)
        
        if not result.get("success"):
            return QueryResponse(success=False, error_message=result.get("error_message"), answer="")

        # 3. 提取 AI 的回答文本
        answer_text = result.get("answer", "未能生成回答。")

        # 4. 清洗文献元数据（这一步决定了网页下面有没有卡片！）
        raw_docs = result.get("source_documents", []) # 假设队友把原始文档放在了这个字段
        formatted_refs = format_references(raw_docs)

        print(f"✅ [Agent] 处理完成，共返回 {len(formatted_refs)} 张文献卡片。")

        # 5. 严格按照 schemas.py 组装并返回
        return QueryResponse(
            success=True,
            intent=intent_info["intent_type"],
            answer=answer_text,
            references=formatted_refs
        )

    except Exception as e:
        print(f"❌ [Agent] 运行崩溃: {e}")
        return QueryResponse(success=False, error_message=str(e), answer="")