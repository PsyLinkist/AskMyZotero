# 说明：
# 本文件负责扫描 Zotero 本地目录中的 PDF 文件，并抽取基础文件元信息。
# 这些扫描结果会被 parser.py 用于解析 PDF，也会被 manifest.py 用于保存目录快照。

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfFileInfo:
    """
    保存单个 PDF 文件的基础元信息。
    这些信息后续可以用于做变更检测和增量索引。
    """
    rel_path: str
    abs_path: Path
    size: int
    mtime: float


def is_pdf_file(file_path: Path) -> bool:
    """
    判断一个路径是否为有效的 PDF 文件。
    这里只检查它是不是普通文件，且后缀名是否为 .pdf。
    """
    return file_path.is_file() and file_path.suffix.lower() == ".pdf"


def scan_pdf_files(zotero_path: Path) -> list[PdfFileInfo]:
    """
    递归扫描 Zotero 目录下的全部 PDF 文件。
    返回的结果按相对路径排序，方便后续做快照比对和调试查看。
    """
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
    """
    打印本次扫描结果摘要。
    当前只输出文件总数，后续也可以扩展输出新增、修改、删除统计。
    """
    print(f"📂 本次共扫描到 {len(pdf_files)} 个 PDF 文件。")