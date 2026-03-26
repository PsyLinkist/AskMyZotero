# 说明：
# 本文件负责保存和读取文献库扫描快照（manifest）。
# 当前版本先记录“有哪些 PDF 文件、文件大小、修改时间、最近扫描时间”，为后续增量同步做准备。

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scanner import PdfFileInfo


def create_empty_manifest() -> dict[str, Any]:
    """
    创建一个空的 manifest 结构。
    当程序第一次运行时，如果本地还没有 manifest.json，就会使用这个默认结构。
    """
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
    """
    从磁盘读取 manifest.json。
    如果文件不存在，则返回一个默认的空 manifest。
    """
    if not manifest_path.exists():
        return create_empty_manifest()

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """
    将 manifest 内容保存到本地 JSON 文件。
    这样下次启动时就能读取到上一次扫描的目录快照。
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_file_records(pdf_files: list[PdfFileInfo]) -> dict[str, dict[str, Any]]:
    """
    将扫描结果转换为适合保存到 JSON 的结构。
    这里以相对路径为 key，保存绝对路径、大小和修改时间等基础信息。
    """
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
    """
    用当前扫描结果更新 manifest 快照。
    当前版本是全量覆盖式更新，后续可以在这里扩展出新增/修改/删除检测逻辑。
    """
    manifest["meta"]["last_scan_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["meta"]["zotero_path"] = str(zotero_path)
    manifest["meta"]["file_count"] = len(pdf_files)
    manifest["files"] = build_file_records(pdf_files)
    return manifest