"""本文件负责扫描 Zotero 存储目录，并汇总发现的 PDF 文件信息。"""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any


@dataclass
class PdfFileInfo:
    """保存单个 PDF 文件的基础元信息。"""
    rel_path: str
    abs_path: Path
    size: int
    mtime: float


def is_pdf_file(file_path: Path) -> bool:
    """判断一个路径是否为有效的 PDF 文件。"""
    return file_path.is_file() and file_path.suffix.lower() == ".pdf"


def scan_pdf_files(zotero_path: Path) -> list[PdfFileInfo]:
    """递归扫描 Zotero 目录下的全部 PDF 文件。"""
    pdf_files: list[PdfFileInfo] = []

    for file_path in zotero_path.rglob("*.pdf"):
        if not is_pdf_file(file_path):
            continue

        stat = file_path.stat()
        rel_path = str(file_path.relative_to(zotero_path)).replace("\\", "/")
        pdf_files.append(
            PdfFileInfo(
                rel_path=rel_path,
                abs_path=file_path.resolve(),
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
        )

    pdf_files.sort(key=lambda x: x.rel_path.lower())
    return pdf_files


def print_scan_summary(pdf_files: list[PdfFileInfo]) -> None:
    """打印本次扫描结果摘要。"""
    print(f"📂 本次共扫描到 {len(pdf_files)} 个 PDF 文件。")


def _pick_zotero_db(storage_path: Path) -> Path | None:
    root = storage_path.parent
    candidates = [
        root / "zotero.sqlite.bak",
        root / "zotero.sqlite.1.bak",
        root / "zotero.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_year(value: Any) -> int | None:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _normalize_title(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", text)
    return text.strip()


def load_attachment_metadata(storage_path: Path) -> dict[str, dict[str, Any]]:
    db_path = _pick_zotero_db(storage_path)
    if db_path is None:
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            ai.itemID AS attachment_item_id,
            ai.key AS attachment_key,
            ia.parentItemID AS parent_item_id,
            pi.key AS parent_key,
            ia.path AS attachment_path
        FROM itemAttachments ia
        JOIN items ai ON ai.itemID = ia.itemID
        LEFT JOIN items pi ON pi.itemID = ia.parentItemID
        WHERE ia.parentItemID IS NOT NULL
          AND ia.path LIKE 'storage:%'
        """
    )
    attachment_rows = cur.fetchall()

    cur.execute(
        """
        SELECT
            id.itemID,
            f.fieldName,
            v.value
        FROM itemData id
        JOIN fields f ON f.fieldID = id.fieldID
        JOIN itemDataValues v ON v.valueID = id.valueID
        WHERE f.fieldName IN ('title', 'date', 'publicationTitle', 'proceedingsTitle', 'DOI', 'abstractNote')
        """
    )
    fields_by_item: dict[int, dict[str, str]] = {}
    for row in cur.fetchall():
        fields_by_item.setdefault(row["itemID"], {})[row["fieldName"]] = row["value"]

    cur.execute(
        """
        SELECT
            ic.itemID,
            c.firstName,
            c.lastName,
            c.fieldMode,
            ic.orderIndex
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        ORDER BY ic.itemID, ic.orderIndex
        """
    )
    creators_by_item: dict[int, list[str]] = {}
    for row in cur.fetchall():
        if row["fieldMode"] == 1:
            name = (row["lastName"] or "").strip()
        else:
            first = (row["firstName"] or "").strip()
            last = (row["lastName"] or "").strip()
            name = " ".join(part for part in [first, last] if part).strip()
        if name:
            creators_by_item.setdefault(row["itemID"], []).append(name)

    metadata_by_attachment: dict[str, dict[str, Any]] = {}
    for row in attachment_rows:
        attachment_key = row["attachment_key"]
        parent_item_id = row["parent_item_id"]
        parent_key = row["parent_key"]
        attachment_path = str(row["attachment_path"] or "")
        filename = attachment_path.split("storage:", 1)[-1].strip()
        rel_path = f"{attachment_key}/{filename}" if attachment_key and filename else ""

        parent_fields = fields_by_item.get(parent_item_id or -1, {})
        venue = parent_fields.get("publicationTitle") or parent_fields.get("proceedingsTitle")
        doi = parent_fields.get("DOI")
        year = _extract_year(parent_fields.get("date"))
        title = parent_fields.get("title")
        normalized_title = _normalize_title(title)
        canonical_paper_id = (
            (f"doi:{doi.lower()}" if doi else None)
            or (f"title_year:{normalized_title}:{year}" if normalized_title and year else None)
            or parent_key
            or attachment_key
            or rel_path
        )

        metadata = {
            "paper_id": canonical_paper_id,
            "attachment_key": attachment_key,
            "parent_item_key": parent_key,
            "paper_title": title,
            "authors": creators_by_item.get(parent_item_id or -1),
            "year": year,
            "venue": venue,
            "doi": doi,
            "abstract": parent_fields.get("abstractNote"),
        }

        if rel_path:
            metadata_by_attachment[rel_path.replace("\\", "/")] = metadata
        if attachment_key:
            metadata_by_attachment[attachment_key] = metadata

    conn.close()
    return metadata_by_attachment
