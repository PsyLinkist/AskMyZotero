from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ==========================================
# 4. 配置读写：RAG 引擎核心参数
# ==========================================
class RagConfigRequest(BaseModel):
    """前端设置页面 POST /api/config 时发送的请求体"""
    zotero_path: str = Field(..., description="Zotero 本地 PDF 文件夹路径")
    api_key: str = Field("", description="OpenAI 风格 API Key，留空则保留现有值")
    base_url: Optional[str] = Field(None, description="API 代理地址，使用官方 OpenAI 留空")
    chat_model: str = Field("gpt-4o-mini", description="聊天模型名称")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding 模型名称")


class RagConfigResponse(BaseModel):
    """GET /api/config 的响应体"""
    success: bool
    message: str = ""
    zotero_path: str = ""
    base_url: Optional[str] = None
    chat_model: str = ""
    embedding_model: str = ""
    api_key_set: bool = Field(False, description="是否已设置 API Key（不返回明文）")

# ==========================================
# 1. 响应：引用文献的详细碎片 (对应前端卡片)
# ==========================================
class EvidenceHit(BaseModel):
    """同一篇文献下的单条检索命中片段（正文）"""
    page: Optional[int] = None
    content: str = ""
    rank: Optional[int] = None


class ReferenceSnippet(BaseModel):
    """对应前端每一个文献卡片所需的数据"""
    title: str = Field(..., description="文献标题")
    authors: str = Field("Unknown", description="作者列表字符串（Zotero 接入后填充）")
    venue: str = Field("N/A", description="发表会议或期刊（Zotero 接入后填充）")
    year: Optional[int] = Field(None, description="发表年份（Zotero 接入后填充）")
    abstract: str = Field(
        "",
        description="预留：Zotero 官方摘要；当前可为空，正文证据见 evidence_snippets",
    )
    source_path: str = Field(..., description="本地文件路径，用于跳转")
    page: Optional[int] = Field(None, description="页码")
    score: float = Field(0.0, description="相关度得分")
    evidence_snippets: List[EvidenceHit] = Field(
        default_factory=list, description="检索命中的正文片段列表，按 rank/页码排序",
    )

# ==========================================
# 2. 响应：返回给前端的完整包
# ==========================================
class QueryResponse(BaseModel):
    success: bool = Field(True)
    error_message: Optional[str] = Field(None)
    
    intent: str = Field("paper_search", description="意图识别结果")
    answer: str = Field(..., description="AI 生成的回复正文")
    
    # 这里变成了刚才定义的详细碎片列表
    references: List[ReferenceSnippet] = Field(default_factory=list)
    
    meta_data: Dict[str, Any] = Field(default_factory=dict)

# ==========================================
# 3. 请求：前端发来的搜索请求
# ==========================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户的问题")
    top_k: int = Field(default=4, description="对应前端的 maxResults")
    # 预留给 Agent 同学做高级过滤 (如年份、顶会限制)
    filters: Dict[str, Any] = Field(default_factory=dict)