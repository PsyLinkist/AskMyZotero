"""Clean API schemas used by the current FastAPI service."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RagConfigRequest(BaseModel):
    zotero_path: str = Field(..., description="Zotero local PDF directory")
    api_key: str = Field("", description="OpenAI-style API key")
    base_url: Optional[str] = Field(None, description="Optional API base URL")
    chat_model: str = Field("gpt-4o-mini", description="Chat model name")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model name")


class RagConfigResponse(BaseModel):
    success: bool
    message: str = ""
    zotero_path: str = ""
    base_url: Optional[str] = None
    chat_model: str = ""
    embedding_model: str = ""
    api_key_set: bool = Field(False, description="Whether API key is configured")


class EvidenceHit(BaseModel):
    page: Optional[int] = None
    content: str = ""
    rank: Optional[int] = None


class ReferenceSnippet(BaseModel):
    title: str
    authors: str = "Unknown"
    venue: str = "N/A"
    year: Optional[int] = None
    abstract: str = ""
    source_path: str
    page: Optional[int] = None
    score: float = 0.0
    evidence_snippets: List[EvidenceHit] = Field(default_factory=list)


class QueryResponse(BaseModel):
    success: bool = True
    error_message: Optional[str] = None
    intent: str = "paper_lookup"
    answer: str
    answer_type: str = "paper_list"
    evidence_summary: List[str] = Field(default_factory=list)
    references: List[ReferenceSnippet] = Field(default_factory=list)
    meta_data: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    filters: Dict[str, Any] = Field(default_factory=dict)
