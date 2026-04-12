"""本文件负责读写和更新本地 PDF 扫描结果的 manifest 快照。"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scanner import PdfFileInfo


def create_empty_manifest() -> dict[str, Any]:
    """创建一个空的 manifest 结构。"""
    return {
        "meta": {
            "version": 1,
            "last_scan_at": None,
            "zotero_path": None,
            "file_count": 0,
        },
        "files": {},
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """从磁盘读取 manifest.json；如果不存在则返回空结构。"""
    if not manifest_path.exists():
        return create_empty_manifest()

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """将 manifest 内容保存到本地 JSON 文件。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_file_records(pdf_files: list[PdfFileInfo]) -> dict[str, dict[str, Any]]:
    """将扫描结果转换为适合写入 JSON 的文件记录结构。"""
    records: dict[str, dict[str, Any]] = {}
    for item in pdf_files:
        records[item.rel_path] = {
            "abs_path": str(item.abs_path),
            "size": item.size,
            "mtime": item.mtime,
        }
    return records


def update_manifest_snapshot(
    manifest: dict[str, Any],
    zotero_path: Path,
    pdf_files: list[PdfFileInfo],
) -> dict[str, Any]:
    """用当前扫描结果更新 manifest 快照。"""
    manifest["meta"]["last_scan_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["meta"]["zotero_path"] = str(zotero_path)
    manifest["meta"]["file_count"] = len(pdf_files)
    manifest["files"] = build_file_records(pdf_files)
    return manifest
