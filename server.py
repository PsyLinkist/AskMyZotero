"""本文件负责提供 FastAPI 服务，以及前端页面、问答接口和配置接口。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn
import yaml
from pathlib import Path
import os
import sys
import shutil

from src.api_models import (
    QueryRequest,
    QueryResponse,
    ReferenceSnippet,
    EvidenceHit,
    RagConfigRequest,
    RagConfigResponse,
)
from src.config import parse_args, resolve_config
from src.qa_agent import ZoteroAgent


def resource_path(name: str) -> Path:
    """兼容 PyInstaller onefile 的资源路径解析。"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / name
    return Path(__file__).resolve().parent / name


def get_runtime_config_path() -> Path:
    """配置文件存放在用户目录，避免 exe 内部不可写。"""
    env_path = os.getenv("ASKMYZOTERO_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    appdata = Path(os.getenv("APPDATA", str(Path.home())))
    return (appdata / "AskMyZotero" / "config.yaml").resolve()


def ensure_runtime_config() -> Path:
    """首次启动时将默认 config.yaml 复制到用户目录。"""
    cfg_path = get_runtime_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists():
        return cfg_path

    template = resource_path("config.yaml")
    if template.exists():
        shutil.copyfile(template, cfg_path)
    else:
        cfg_path.write_text("", encoding="utf-8")
    return cfg_path


CONFIG_FILE = ensure_runtime_config()
os.environ.setdefault("ASKMYZOTERO_CONFIG", str(CONFIG_FILE))


def _read_config_dict() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_sufficient_for_agent(cfg: dict) -> bool:
    """与 GET /api/config 的“可检索”条件对齐：路径 + 已保存的 API Key。"""
    if not cfg:
        return False
    zp = str(cfg.get("zotero_path", "")).strip()
    key = str(cfg.get("api_key", "")).strip()
    return bool(zp and key)


app = FastAPI(title="Zotero RAG API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_main_ui():
    """主页；避免 EXE 下 file:// 指向 Temp\\_MEI* 时相对跳转失效，统一走 HTTP。"""
    p = resource_path("zotero_rag_ui.html")
    return FileResponse(str(p), media_type="text/html; charset=utf-8")


@app.get("/zotero_rag_ui.html")
async def redirect_legacy_main():
    """旧链接兼容，避免重复一份 FileResponse。"""
    return RedirectResponse(url="/", status_code=302)


@app.get("/settings.html")
async def serve_settings_ui():
    p = resource_path("settings.html")
    return FileResponse(str(p), media_type="text/html; charset=utf-8")


# 全局变量，防止重复加载模型
GLOBAL_CONTEXT: dict = {}


def init_agent() -> tuple[bool, str]:
    """按当前配置初始化 Agent；失败时返回错误信息，不中断服务。"""
    args = parse_args()
    config = resolve_config(args)
    GLOBAL_CONTEXT["agent"] = ZoteroAgent(config)
    GLOBAL_CONTEXT["init_error"] = ""
    return True, "后端引擎已就绪"


@app.on_event("startup")
def startup():
    """服务启动：若 config.yaml 已具备路径与 API Key，则自动初始化引擎（无需再点一次保存）。"""
    print("⏳ API 服务启动完成…")
    GLOBAL_CONTEXT["agent"] = None
    GLOBAL_CONTEXT["init_error"] = ""
    cfg = _read_config_dict()
    if not _config_sufficient_for_agent(cfg):
        print("未检测到完整本地配置（zotero_path + api_key），等待通过设置页保存或 /api/init 初始化。")
        return
    try:
        ok, msg = init_agent()
        if ok:
            print("✅ 已根据现有 config.yaml 自动初始化后端引擎。")
        else:
            GLOBAL_CONTEXT["init_error"] = msg or "初始化失败"
            print(f"⚠️ 自动初始化未成功：{msg}")
    except Exception as e:
        GLOBAL_CONTEXT["agent"] = None
        GLOBAL_CONTEXT["init_error"] = str(e)
        print(f"⚠️ 自动初始化失败：{e}")


@app.get("/health")
async def health_check():
    """对应前端『测试连接』按钮"""
    ready = GLOBAL_CONTEXT.get("agent") is not None
    return {
        "status": "ok",
        "message": "Zotero RAG Server is running",
        "ready": ready,
        "init_error": GLOBAL_CONTEXT.get("init_error", ""),
    }


@app.post("/api/ask", response_model=QueryResponse)
async def ask_endpoint(request: QueryRequest):
    agent: ZoteroAgent = GLOBAL_CONTEXT.get("agent")
    if not agent:
        return QueryResponse(
            success=False,
            error_message="后端尚未初始化。请先在设置页保存配置，再返回主页面搜索。",
            answer="",
        )

    try:
        result = agent.handle_query(request.query, request.top_k)

        if not result["success"]:
            return QueryResponse(
                success=False,
                error_message=result.get("error", {}).get("message", "Unknown error"),
                answer=""
            )

        paper_items = result.get("papers", [])
        if paper_items:
            references = [
                ReferenceSnippet(
                    title=p.get("title", "unknown"),
                    authors=", ".join(p.get("authors") or []) if isinstance(p.get("authors"), list) else (p.get("authors") or "Unknown"),
                    venue=p.get("venue") or "N/A",
                    year=p.get("year"),
                    abstract="；".join(p.get("match_reason", [])),
                    source_path=p.get("source_path", p.get("title", "unknown")),
                    page=(p.get("evidences", [{}])[0].get("page") if p.get("evidences") else p.get("page")),
                    score=p.get("score", 0.0),
                    evidence_snippets=[
                        EvidenceHit(
                            page=h.get("page"),
                            content=h.get("raw_text", h.get("text", h.get("content", ""))),
                            rank=h.get("rank"),
                        )
                        for h in p.get("evidences", p.get("evidence_snippets", []))
                    ],
                )
                for p in paper_items
            ]
        else:
            references = [
                ReferenceSnippet(
                    title=(
                        s.get("paper_title")
                        or s.get("title")
                        or Path(str(s.get("source_path", s["source"]))).name
                    ),
                    abstract="",
                    source_path=s.get("source_path", s["source"]),
                    page=s.get("page"),
                    evidence_snippets=[
                        EvidenceHit(
                            page=s.get("page"),
                            content=s.get("raw_text", s.get("text", s.get("content", ""))),
                            rank=s.get("rank"),
                        )
                    ],
                )
                for s in result.get("snippets", [])
            ]

        return QueryResponse(
            success=True,
            answer=result["answer"],
            intent=result.get("intent", "paper_lookup"),
            answer_type=result.get("answer_type", "paper_list"),
            evidence_summary=result.get("evidence_summary", []),
            references=references,
            meta_data={
                "top_k_used": result.get("top_k_used"),
                "status": result.get("status"),
                "confidence": result.get("confidence"),
                "debug": result.get("debug", {}),
            },
        )
    except Exception as e:
        return QueryResponse(success=False, error_message=str(e), answer="")


@app.get("/api/config", response_model=RagConfigResponse)
async def get_config():
    """返回当前 config.yaml 中的核心 RAG 配置（API Key 不明文返回）"""
    if not CONFIG_FILE.exists():
        return RagConfigResponse(success=False, message="config.yaml 不存在")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return RagConfigResponse(
        success=True,
        zotero_path=str(cfg.get("zotero_path", "")),
        base_url=cfg.get("base_url"),
        chat_model=cfg.get("chat_model", ""),
        embedding_model=cfg.get("embedding_model", ""),
        api_key_set=bool(cfg.get("api_key", ""))
    )


@app.post("/api/config", response_model=RagConfigResponse)
async def save_config(request: RagConfigRequest):
    """将设置页面提交的核心参数写入 config.yaml，并尝试重新初始化 Agent"""
    existing: dict = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    existing["zotero_path"] = request.zotero_path
    existing["base_url"] = request.base_url or existing.get("base_url")
    existing["chat_model"] = request.chat_model
    existing["embedding_model"] = request.embedding_model
    if request.api_key.strip():
        existing["api_key"] = request.api_key.strip()

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, allow_unicode=True, default_flow_style=False)

    try:
        ok, msg = init_agent()
        if ok:
            return RagConfigResponse(success=True, message="配置已保存，引擎已初始化")
        return RagConfigResponse(success=False, message=msg)
    except Exception as e:
        GLOBAL_CONTEXT["agent"] = None
        GLOBAL_CONTEXT["init_error"] = str(e)
        return RagConfigResponse(success=False, message=f"配置已写入，但引擎初始化失败：{e}")


@app.post("/api/init", response_model=RagConfigResponse)
async def init_engine():
    """手动触发引擎初始化（可在前端配置完成后调用）。"""
    try:
        ok, msg = init_agent()
        if ok:
            return RagConfigResponse(success=True, message=msg)
        return RagConfigResponse(success=False, message=msg)
    except Exception as e:
        GLOBAL_CONTEXT["agent"] = None
        GLOBAL_CONTEXT["init_error"] = str(e)
        return RagConfigResponse(success=False, message=f"引擎初始化失败：{e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
