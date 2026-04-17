"""Section-aware parser with paragraph chunking and neighbor context windows."""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from src.config import AppConfig
from src.scanner import PdfFileInfo, load_attachment_metadata, scan_pdf_files


CHUNK_STRATEGY_VERSION = "section_paragraph_v3"

SECTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    ("introduction", ("introduction", "1 introduction")),
    ("method", ("method", "methods", "methodology", "approach", "model")),
    ("experiment", ("experiment", "experiments", "evaluation", "results")),
    ("conclusion", ("conclusion", "conclusions", "discussion")),
    ("references", ("references", "bibliography")),
]


def guess_section(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    if not normalized:
        return None
    for section_name, patterns in SECTION_PATTERNS:
        for pattern in patterns:
            if re.search(rf"(^|\W){re.escape(pattern)}(\W|$)", normalized):
                return section_name
    return None


def map_chunk_type(section: str | None) -> str:
    if section in {"abstract", "introduction", "method", "experiment", "conclusion", "references"}:
        return str(section)
    return "body"


def build_chunk_id(paper_id: str, page_start: int | None, page_end: int | None, text: str) -> str:
    normalized = " ".join((text or "").split()).lower().encode("utf-8", errors="ignore")
    text_hash = hashlib.sha1(normalized).hexdigest()[:12]
    page_start_text = str(page_start) if page_start is not None else "na"
    page_end_text = str(page_end) if page_end is not None else "na"
    return f"{paper_id}#p{page_start_text}-{page_end_text}#{text_hash}"


def _clean_page_text(text: str) -> str:
    cleaned = str(text or "").replace("\r", "\n")
    cleaned = re.sub(r"-\n(?=[a-zA-Z])", "", cleaned)
    cleaned = re.sub(r"Authorized licensed use limited to:.*?(?:\n|$)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Downloaded on .*? from IEEE Xplore\. Restrictions apply\.", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"IEEE TRANSACTIONS ON [A-Z ,.\-0-9]+", " ", cleaned)
    cleaned = re.sub(r"\bVOL\.\s*\d+,\s*NO\.\s*\d+,\s*[A-Z]+\s*\d{4}\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{3,5}\s+IEEE\b", " ", cleaned)
    cleaned = re.sub(r"\bTABLE\s+[IVXLC]+\b", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_paragraph_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", normalized)
    return normalized


def _split_page_into_paragraphs(text: str) -> list[str]:
    cleaned = _clean_page_text(text)
    if not cleaned:
        return []
    rough_parts = re.split(r"\n\s*\n+", cleaned)
    paragraphs: list[str] = []
    for part in rough_parts:
        candidate = re.sub(r"\s+", " ", part).strip()
        if not candidate:
            continue
        if paragraphs and len(candidate) < 80:
            paragraphs[-1] = f"{paragraphs[-1]} {candidate}".strip()
        else:
            paragraphs.append(candidate)
    deduped: list[str] = []
    last_sig = ""
    for paragraph in paragraphs:
        sig = _normalize_paragraph_signature(paragraph)
        if sig and sig == last_sig:
            continue
        deduped.append(paragraph)
        last_sig = sig
    return deduped


def _dedupe_join_texts(*texts: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for paragraph in _split_page_into_paragraphs(text):
            sig = _normalize_paragraph_signature(paragraph)
            if not sig or sig in seen:
                continue
            seen.add(sig)
            merged.append(paragraph)
    return "\n\n".join(merged).strip()

# yzx update for PO
# def load_pdf_documents_from_files(
#     pdf_files: list[PdfFileInfo],
#     attachment_metadata: dict[str, dict] | None = None,
# ) -> list[Document]:
#     docs: list[Document] = []
#     for item in tqdm(pdf_files, desc="📄 解析 PDF 进度"):
#         try:
#             loader = PyPDFLoader(str(item.abs_path))
#             file_docs = loader.load()
#             for doc in file_docs:
#                 doc.page_content = _clean_page_text(doc.page_content)
#                 doc.metadata["source"] = str(item.abs_path)
#                 doc.metadata["rel_path"] = item.rel_path
#                 doc.metadata["file_size"] = item.size
#                 doc.metadata["file_mtime"] = item.mtime
#                 meta = (attachment_metadata or {}).get(item.rel_path) or (attachment_metadata or {}).get(item.rel_path.split("/", 1)[0])
#                 if meta:
#                     doc.metadata["paper_id"] = meta.get("paper_id")
#                     doc.metadata["paper_title"] = meta.get("paper_title")
#                     doc.metadata["authors"] = meta.get("authors")
#                     doc.metadata["year"] = meta.get("year")
#                     doc.metadata["venue"] = meta.get("venue")
#                     doc.metadata["doi"] = meta.get("doi")
#                     doc.metadata["attachment_key"] = meta.get("attachment_key")
#                     doc.metadata["parent_item_key"] = meta.get("parent_item_key")
#             docs.extend(file_docs)
#         except Exception as exc:
#             print(f"⚠️ 跳过解析失败的文件: {item.rel_path}，原因: {exc}")
#     if not docs:
#         raise RuntimeError("未读取到任何 PDF 内容，请检查 Zotero storage 路径是否正确。")
#     return docs

def load_pdf_documents_from_files(
    pdf_files: list[PdfFileInfo],
    attachment_metadata: dict[str, dict] | None = None,
    progress_callback: Any = None,  # [新增] 进度回调函数
) -> list[Document]:
    docs: list[Document] = []
    total_files = len(pdf_files) # [新增] 获取总文件数用于计算进度百分比
    
    for idx, item in enumerate(tqdm(pdf_files, desc="📄 解析 PDF 进度")):
        # --- [新增] 触发进度回调 ---
        if progress_callback:
            progress_callback(
                stage="parsing",
                current=idx + 1,
                total=total_files,
                message=f"正在解析: {item.rel_path}"
            )
        # --------------------------
        
        try:
            loader = PyPDFLoader(str(item.abs_path))
            file_docs = loader.load()
            for doc in file_docs:
                doc.page_content = _clean_page_text(doc.page_content)
                doc.metadata["source"] = str(item.abs_path)
                doc.metadata["rel_path"] = item.rel_path
                doc.metadata["file_size"] = item.size
                doc.metadata["file_mtime"] = item.mtime
                meta = (attachment_metadata or {}).get(item.rel_path) or (attachment_metadata or {}).get(item.rel_path.split("/", 1)[0])
                if meta:
                    doc.metadata["paper_id"] = meta.get("paper_id")
                    doc.metadata["paper_title"] = meta.get("paper_title")
                    doc.metadata["authors"] = meta.get("authors")
                    doc.metadata["year"] = meta.get("year")
                    doc.metadata["venue"] = meta.get("venue")
                    doc.metadata["doi"] = meta.get("doi")
                    doc.metadata["attachment_key"] = meta.get("attachment_key")
                    doc.metadata["parent_item_key"] = meta.get("parent_item_key")
            docs.extend(file_docs)
        except Exception as exc:
            print(f"⚠️ 跳过解析失败的文件: {item.rel_path}，原因: {exc}")
    if not docs:
        raise RuntimeError("未读取到任何 PDF 内容，请检查 Zotero storage 路径是否正确。")
    return docs

# yzx update for PO
# def load_pdf_documents(zotero_path: str, attachment_metadata: dict[str, dict] | None = None) -> list[Document]:
#     print(f"🟡 正在扫描目录: {zotero_path}")
#     print("⏳ 正在读取 PDF 文件，请稍候...")
#     pdf_files = scan_pdf_files(zotero_path)
#     docs = load_pdf_documents_from_files(pdf_files, attachment_metadata=attachment_metadata)
#     print(f"✅ PDF 读取完毕！共提取了 {len(docs)} 页文献内容。")
#     return docs

def load_pdf_documents(
    zotero_path: str, 
    attachment_metadata: dict[str, dict] | None = None,
    progress_callback: Any = None  # [新增] 接收回调
) -> list[Document]:
    print(f"🟡 正在扫描目录: {zotero_path}")
    print("⏳ 正在读取 PDF 文件，请稍候...")
    pdf_files = scan_pdf_files(zotero_path)
    # [修改] 将回调传给下游
    docs = load_pdf_documents_from_files(
        pdf_files, 
        attachment_metadata=attachment_metadata, 
        progress_callback=progress_callback
    )
    print(f"✅ PDF 读取完毕！共提取了 {len(docs)} 页文献内容。")
    return docs

def _paragraph_records_from_docs(docs: list[Document]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last_section_by_source: dict[str, str | None] = {}
    for doc in docs:
        metadata = dict(doc.metadata or {})
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        page_1based = page + 1 if isinstance(page, int) else None
        paragraphs = _split_page_into_paragraphs(doc.page_content)
        guessed_page_section = guess_section(doc.page_content)
        current_section = guessed_page_section or last_section_by_source.get(source)
        for paragraph in paragraphs:
            paragraph_section = guess_section(paragraph) or current_section
            if paragraph_section:
                current_section = paragraph_section
                last_section_by_source[source] = paragraph_section
            records.append(
                {
                    "text": paragraph,
                    "section": current_section,
                    "metadata": {
                        **metadata,
                        "page_1based": page_1based,
                    },
                }
            )
    return records


def _build_chunk_document(parts: list[dict[str, Any]]) -> Document:
    first = parts[0]
    last = parts[-1]
    first_meta = dict(first["metadata"])
    rel_path = str(first_meta.get("rel_path") or Path(str(first_meta.get("source", "unknown"))).name)
    paper_title = first_meta.get("paper_title") or Path(rel_path).stem
    page_start = first_meta.get("page_1based")
    page_end = last["metadata"].get("page_1based")
    content = "\n\n".join(part["text"] for part in parts).strip()
    section = first.get("section")
    metadata = {
        **first_meta,
        "source_path": str(first_meta.get("source", "unknown")),
        "source": str(first_meta.get("source", "unknown")),
        "rel_path": rel_path,
        "paper_id": first_meta.get("paper_id") or rel_path,
        "paper_title": paper_title,
        "section": section,
        "chunk_type": map_chunk_type(section),
        "page_start": page_start,
        "page_end": page_end,
        "page": first_meta.get("page"),
        "raw_text": content,
        "paragraph_count": len(parts),
        "chunk_strategy": CHUNK_STRATEGY_VERSION,
    }
    metadata["chunk_id"] = build_chunk_id(str(metadata["paper_id"]), page_start, page_end, content)
    return Document(page_content=content, metadata=metadata)


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    print("✂️ 正在按 section + 段落切割文献文本...")
    if chunk_overlap:
        print("ℹ️ 已启用相邻块上下文窗口，切块阶段忽略 chunk_overlap。")
    records = _paragraph_records_from_docs(docs)
    splits: list[Document] = []
    current_parts: list[dict[str, Any]] = []
    current_source: str | None = None
    current_section: str | None = None
    current_size = 0

    def flush_current() -> None:
        nonlocal current_parts, current_size, current_source, current_section
        if current_parts:
            splits.append(_build_chunk_document(current_parts))
        current_parts = []
        current_size = 0
        current_source = None
        current_section = None

    for record in records:
        source = str(record["metadata"].get("source", "unknown"))
        section = record.get("section")
        paragraph = record["text"]
        paragraph_size = len(paragraph)
        should_flush = False
        if current_parts and source != current_source:
            should_flush = True
        elif current_parts and section != current_section:
            should_flush = True
        elif current_parts and current_size + paragraph_size > chunk_size:
            should_flush = True
        if should_flush:
            flush_current()
        if not current_parts:
            current_source = source
            current_section = section
        current_parts.append(record)
        current_size += paragraph_size
    flush_current()
    if not splits:
        raise RuntimeError("文本切块结果为空，请检查 PDF 内容是否可解析。")
    print(f"✅ 文本切割完成！共生成 {len(splits)} 个文本块。")
    return splits


def enrich_split_metadata(splits: list[Document], attachment_metadata: dict[str, dict] | None = None) -> list[Document]:
    for idx, split in enumerate(splits, start=1):
        metadata = split.metadata or {}
        source_path = str(metadata.get("source", "unknown"))
        file_name = Path(source_path).name if source_path != "unknown" else "unknown"
        rel_path = str(metadata.get("rel_path") or file_name)
        zotero_meta = (attachment_metadata or {}).get(rel_path) or (attachment_metadata or {}).get(rel_path.split("/", 1)[0])
        paper_title = Path(rel_path).stem if rel_path else Path(file_name).stem
        section = metadata.get("section") or guess_section(split.page_content)
        chunk_type = metadata.get("chunk_type") or map_chunk_type(section)
        page = metadata.get("page")
        page_1based = metadata.get("page_1based")
        if page_1based is None and isinstance(page, int):
            page_1based = page + 1
        metadata["source_path"] = source_path
        metadata["source"] = source_path
        metadata["paper_id"] = (zotero_meta or {}).get("paper_id") or metadata.get("paper_id") or rel_path
        metadata["paper_title"] = (zotero_meta or {}).get("paper_title") or metadata.get("paper_title") or paper_title
        metadata["authors"] = (zotero_meta or {}).get("authors") or metadata.get("authors")
        metadata["year"] = (zotero_meta or {}).get("year") or metadata.get("year")
        metadata["venue"] = (zotero_meta or {}).get("venue") or metadata.get("venue")
        metadata["doi"] = (zotero_meta or {}).get("doi") or metadata.get("doi")
        metadata["attachment_key"] = (zotero_meta or {}).get("attachment_key") or metadata.get("attachment_key")
        metadata["parent_item_key"] = (zotero_meta or {}).get("parent_item_key") or metadata.get("parent_item_key")
        metadata["rel_path"] = rel_path
        metadata["section"] = section
        metadata["chunk_type"] = chunk_type
        metadata["file_name"] = metadata.get("file_name", file_name)
        metadata["page"] = page if isinstance(page, int) else metadata.get("page")
        metadata["page_1based"] = page_1based
        metadata["page_start"] = metadata.get("page_start", page_1based)
        metadata["page_end"] = metadata.get("page_end", page_1based)
        metadata["raw_text"] = metadata.get("raw_text") or split.page_content or ""
        metadata["chunk_id"] = metadata.get("chunk_id") or build_chunk_id(
            str(metadata["paper_id"]),
            metadata["page_start"],
            metadata["page_end"],
            metadata["raw_text"],
        )
        metadata["chunk_rank"] = idx
        metadata["chunk_strategy"] = metadata.get("chunk_strategy") or "legacy"
        split.metadata = metadata

    grouped: dict[str, list[Document]] = {}
    for split in splits:
        source = str(split.metadata.get("source", "unknown"))
        grouped.setdefault(source, []).append(split)

    for source_splits in grouped.values():
        total = len(source_splits)
        for index, split in enumerate(source_splits):
            prev_doc = source_splits[index - 1] if index > 0 else None
            next_doc = source_splits[index + 1] if index + 1 < total else None
            split.metadata["chunk_local_index"] = index + 1
            split.metadata["chunk_local_total"] = total
            split.metadata["prev_chunk_id"] = prev_doc.metadata.get("chunk_id") if prev_doc else None
            split.metadata["next_chunk_id"] = next_doc.metadata.get("chunk_id") if next_doc else None
            split.metadata["context_before"] = prev_doc.page_content if prev_doc else ""
            split.metadata["context_after"] = next_doc.page_content if next_doc else ""
            split.metadata["context_window"] = _dedupe_join_texts(
                split.metadata["context_before"],
                split.page_content,
                split.metadata["context_after"],
            )
    return splits


def load_or_create_splits(config: AppConfig) -> list[Document]:
    attachment_metadata = load_attachment_metadata(config.zotero_path)
    if config.splits_cache_path.exists():
        print(f"🟢 发现文本块缓存，直接读取: {config.splits_cache_path}")
        with open(config.splits_cache_path, "rb") as f:
            splits = pickle.load(f)
        current_strategy = None
        if splits:
            current_strategy = (splits[0].metadata or {}).get("chunk_strategy")
        if current_strategy == CHUNK_STRATEGY_VERSION:
            splits = enrich_split_metadata(splits, attachment_metadata=attachment_metadata)
            print(f"✅ 缓存读取成功，共 {len(splits)} 个文本块。")
            return splits
        print("♻️ 检测到旧版切块缓存，正在按新策略重建...")


    # yzx update for PO [新增] 安全地从 config 中获取 callback
    callback = getattr(config, 'progress_callback', None)

    # [修改] 将回调传给下游
    docs = load_pdf_documents(
        config.zotero_path, 
        attachment_metadata=attachment_metadata,
        progress_callback=callback
    )
    splits = split_documents(docs, config.chunk_size, config.chunk_overlap)
    splits = enrich_split_metadata(splits, attachment_metadata=attachment_metadata)
    
    # docs = load_pdf_documents(config.zotero_path, attachment_metadata=attachment_metadata)
    # splits = split_documents(docs, config.chunk_size, config.chunk_overlap)
    # splits = enrich_split_metadata(splits, attachment_metadata=attachment_metadata)
    print("💾 正在保存文本块缓存...")
    with open(config.splits_cache_path, "wb") as f:
        pickle.dump(splits, f)
    print("✅ 文本块缓存保存成功。")
    return splits
