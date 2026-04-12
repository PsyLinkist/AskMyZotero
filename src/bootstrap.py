"""本文件负责准备启动阶段需要的基础数据，例如当前 PDF 清单快照。"""

from src.scanner import scan_pdf_files, print_scan_summary
from src.manifest import load_manifest, save_manifest, update_manifest_snapshot


def prepare_manifest_snapshot(config) -> None:
    """
    扫描当前文献目录，并把结果写入 manifest.json。
    供 Agent/Server/CLI 在启动阶段复用。
    """
    pdf_files = scan_pdf_files(config.zotero_path)
    print_scan_summary(pdf_files)

    manifest = load_manifest(config.manifest_path)
    manifest = update_manifest_snapshot(manifest, config.zotero_path, pdf_files)
    save_manifest(config.manifest_path, manifest)

    print(f"📝 Manifest 已更新: {config.manifest_path}")
