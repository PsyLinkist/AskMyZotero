"""本文件负责读写和更新本地 PDF 扫描结果的 manifest 快照。"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scanner import PdfFileInfo, print_scan_summary, scan_pdf_files


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


def diff_manifest_files(
    old_files: dict[str, dict[str, Any]] | None,
    new_files: dict[str, dict[str, Any]] | None,
) -> dict[str, list[str]]:
    """Compare two manifest file maps and classify path-level changes."""
    old_files = old_files or {}
    new_files = new_files or {}
    old_paths = set(old_files)
    new_paths = set(new_files)
    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)
    modified: list[str] = []
    for rel_path in sorted(old_paths & new_paths):
        old_record = old_files.get(rel_path) or {}
        new_record = new_files.get(rel_path) or {}
        if old_record.get("size") != new_record.get("size") or old_record.get("mtime") != new_record.get("mtime"):
            modified.append(rel_path)
    unchanged = sorted(new_paths - set(added) - set(modified))
    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "unchanged": unchanged,
    }


def prepare_manifest_snapshot(config) -> None:
    """扫描当前文献目录，并将结果写入 manifest.json。"""
    pdf_files = scan_pdf_files(config.zotero_path)
    print_scan_summary(pdf_files)
    manifest = load_manifest(config.manifest_path)
    manifest = update_manifest_snapshot(manifest, config.zotero_path, pdf_files)
    save_manifest(config.manifest_path, manifest)
    print(f"📘 Manifest 已更新: {config.manifest_path}")
