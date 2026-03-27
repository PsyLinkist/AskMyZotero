from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ==========================================
# 1. 响应：引用文献的详细碎片 (对应前端卡片)
# ==========================================
class ReferenceSnippet(BaseModel):
    """对应前端每一个文献卡片所需的数据"""
    title: str = Field(..., description="文献标题")
    authors: str = Field("Unknown", description="作者列表字符串")
    venue: str = Field("N/A", description="发表会议或期刊")
    year: Optional[int] = Field(None, description="发表年份")
    abstract: str = Field(..., description="原文片段或摘要内容")
    source_path: str = Field(..., description="本地文件路径，用于跳转")
    page: Optional[int] = Field(None, description="页码")
    score: float = Field(0.0, description="相关度得分")

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