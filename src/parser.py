# 说明：
# 本文件负责把本地 PDF 解析成 LangChain 文档，并进一步切分为文本块。
# 同时它也负责管理 splits 的本地缓存，从而避免每次启动都重新解析全部 PDF。

import pickle
import re
import hashlib
from pathlib import Path

from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig
from src.metadata_store import load_attachment_metadata
from src.scanner import PdfFileInfo, scan_pdf_files


SECTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    ("introduction", ("introduction", "1 introduction")),
    ("method", ("method", "methods", "methodology", "approach", "model")),
    ("experiment", ("experiment", "experiments", "evaluation", "results")),
    ("conclusion", ("conclusion", "conclusions", "discussion")),
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
    if section in {"abstract", "introduction", "method", "experiment", "conclusion"}:
        return str(section)
    return "body"


def build_chunk_id(paper_id: str, page_start: int | None, page_end: int | None, text: str) -> str:
    normalized = " ".join((text or "").split()).lower().encode("utf-8", errors="ignore")
    text_hash = hashlib.sha1(normalized).hexdigest()[:12]
    page_start_text = str(page_start) if page_start is not None else "na"
    page_end_text = str(page_end) if page_end is not None else "na"
    return f"{paper_id}#p{page_start_text}-{page_end_text}#{text_hash}"


def load_pdf_documents_from_files(pdf_files: list[PdfFileInfo], attachment_metadata: dict[str, dict] | None = None):
    """
    按文件列表逐个读取 PDF 内容，并合并成统一的 LangChain 文档列表。
    这里会尽量保留每个 PDF 的来源元信息，便于后续检索展示和增量更新。
    """
    docs = []

    for item in tqdm(pdf_files, desc="📄 解析 PDF 进度"):
        try:
            loader = PyPDFLoader(str(item.abs_path))
            file_docs = loader.load()

            for doc in file_docs:
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
        except Exception as e:
            print(f"⚠️ 跳过解析失败的文件: {item.rel_path}，原因: {e}")

    if not docs:
        raise RuntimeError("未读取到任何 PDF 内容，请检查 Zotero storage 路径是否正确。")

    return docs


def load_pdf_documents(zotero_path, attachment_metadata: dict[str, dict] | None = None):
    """
    递归扫描 Zotero 目录，并读取所有 PDF 的内容。
    这个函数是“目录路径 -> 文档列表”的总入口，供切块和建索引阶段复用。
    """
    print(f"🟡 正在扫描目录: {zotero_path}")
    print("⏳ 正在读取 PDF 文件，请稍候...")

    pdf_files = scan_pdf_files(zotero_path)
    docs = load_pdf_documents_from_files(pdf_files, attachment_metadata=attachment_metadata)

    print(f"✅ PDF 读取完毕！共提取了 {len(docs)} 页文献内容。")
    return docs


def split_documents(docs, chunk_size: int, chunk_overlap: int):
    """
    将长文档切分为适合嵌入和检索的文本块。
    这样既能降低单次嵌入压力，也能提高检索粒度。
    """
    print("✂️ 正在切割文献文本...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    if not splits:
        raise RuntimeError("文本切块结果为空，请检查 PDF 内容是否可解析。")

    print(f"✅ 文本切割完成！共生成 {len(splits)} 个文本块。")
    return splits


def enrich_split_metadata(splits, attachment_metadata: dict[str, dict] | None = None):
    """
    统一补齐检索与前端展示需要的 metadata，确保历史缓存/新切块格式一致。
    """
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
        page_1based = page + 1 if isinstance(page, int) else None

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
        metadata["chunk_id"] = build_chunk_id(
            str(metadata["paper_id"]),
            metadata["page_start"],
            metadata["page_end"],
            split.page_content or "",
        )
        metadata["chunk_rank"] = idx

        split.metadata = metadata

    return splits


def load_or_create_splits(config: AppConfig):
    """
    优先从本地缓存中读取文本块；如果缓存不存在，则重新解析 PDF 并切块。
    这一步能显著减少重复启动时的 PDF 解析耗时。
    """
    attachment_metadata = load_attachment_metadata(config.zotero_path)

    if config.splits_cache_path.exists():
        print(f"🟢 发现文本块缓存，直接读取: {config.splits_cache_path}")
        with open(config.splits_cache_path, "rb") as f:
            splits = pickle.load(f)
        splits = enrich_split_metadata(splits, attachment_metadata=attachment_metadata)
        print(f"✅ 缓存读取成功，共 {len(splits)} 个文本块。")
        return splits

    docs = load_pdf_documents(config.zotero_path, attachment_metadata=attachment_metadata)
    splits = split_documents(docs, config.chunk_size, config.chunk_overlap)
    splits = enrich_split_metadata(splits, attachment_metadata=attachment_metadata)

    print("💾 正在保存文本块缓存...")
    with open(config.splits_cache_path, "wb") as f:
        pickle.dump(splits, f)
    print("✅ 文本块缓存保存成功。")

    return splits
