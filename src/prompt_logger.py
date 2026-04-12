"""本文件负责保存提示词、检索结果和回答内容，便于后续排查问题。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def save_prompt_log(
    work_dir: Path,
    *,
    query: str,
    query_bundle: Any,
    chunks: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    answer: str,
    status: str,
    debug: dict[str, Any] | None = None,
) -> Path:
    logs_dir = Path(work_dir) / "logs" / "prompt_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = logs_dir / f"{timestamp}.json"

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "query_bundle": query_bundle.to_dict() if hasattr(query_bundle, "to_dict") else query_bundle,
        "retrieved_chunk_count": len(chunks),
        "retrieved_chunks": chunks,
        "paper_count": len(papers),
        "papers": papers,
        "answer": answer,
        "status": status,
        "debug": debug or {},
    }

    log_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return log_path
