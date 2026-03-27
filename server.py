from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入上面定义的协议
from src.schemas import QueryRequest, QueryResponse, ReferenceSnippet
# 导入队友的底层逻辑
from src.config import parse_args, resolve_config
from src.indexer import get_vectorstore, create_chat_chain, api_ask_once
from main import prepare_manifest_snapshot

app = FastAPI(title="Zotero RAG API Server")

# 解决跨域，否则前端 HTML 连不上
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量，防止重复加载模型
GLOBAL_CONTEXT = {}

@app.on_event("startup")
def startup():
    """服务启动时，一次性把库和模型加载进内存"""
    print("⏳ 正在初始化后端引擎...")
    args = parse_args() 
    config = resolve_config(args)
    
    # 扫描并加载
    prepare_manifest_snapshot(config)
    vectorstore = get_vectorstore(config)
    rag_chain = create_chat_chain(config, vectorstore)
    
    GLOBAL_CONTEXT["rag_chain"] = rag_chain

    print("✅ 后端引擎已就绪 (Port: 8000)")

@app.get("/health")
async def health_check():
    """对应前端『测试连接』按钮"""
    return {"status": "ok", "message": "Zotero RAG Server is running"}

@app.post("/api/ask", response_model=QueryResponse)
async def ask_endpoint(request: QueryRequest):
    chain = GLOBAL_CONTEXT.get("rag_chain")
    if not chain:
        return QueryResponse(success=False, error_message="Server not initialized", answer="")

    try:
        # 调用 indexer 的执行函数
        # 注意：这里需要你确保 indexer.py 里的 api_ask_once 能返回 docs 的 metadata
        result = api_ask_once(chain, request.query)
        
        if not result["success"]:
            return QueryResponse(success=False, error_message=result.get("error_message"), answer="")

        return QueryResponse(
            success=True,
            answer=result["answer"],
            # 这里的 references 需要从底层 result 里提取 metadata 组装
            references=result.get("references", []), 
            intent="paper_search"
        )
    except Exception as e:
        return QueryResponse(success=False, error_message=str(e), answer="")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    