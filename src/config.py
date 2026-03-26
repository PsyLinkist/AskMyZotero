# 说明：
# 本文件负责统一管理命令行参数、环境变量读取和运行配置对象。
# 其他模块只依赖 AppConfig，不再各自维护零散的路径、模型和 API 参数。

import os
import getpass
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    """
    统一保存程序运行所需的全部配置。
    这样可以避免在多个模块中重复传递零散参数。
    """
    zotero_path: Path
    work_dir: Path
    index_name: str
    db_save_path: Path
    splits_cache_path: Path
    manifest_path: Path

    api_key: str
    base_url: Optional[str]

    chat_model: str
    embedding_model: str

    chunk_size: int
    chunk_overlap: int
    top_k: int

    temperature: float
    max_completion_tokens: Optional[int]

    rebuild: bool
    question: Optional[str]
    no_proxy: Optional[str]


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    这里统一收集路径、模型、索引、对话模式等启动配置，供主程序直接使用。
    """
    parser = argparse.ArgumentParser(
        description="AskMyZotero - 基于本地 Zotero PDF 文献库的 RAG 命令行助手",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基础配置
    parser.add_argument("--zotero-path", type=str, default=None, help="Zotero 本地 storage 路径")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI 风格 API Key")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI 风格 BASE_URL，例如 https://api.openai.com/v1")
    parser.add_argument("--chat-model", type=str, default="gpt-4o-mini", help="聊天模型名称")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="嵌入模型名称")

    # 索引与缓存
    parser.add_argument("--work-dir", type=str, default=".askmyzotero", help="工作目录")
    parser.add_argument("--index-name", type=str, default="default", help="索引名称，用于区分不同配置")
    parser.add_argument("--rebuild", action="store_true", help="强制删除旧索引与缓存并重建")

    # 文本切分 / 检索参数
    parser.add_argument("--chunk-size", type=int, default=1000, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="文本块重叠")
    parser.add_argument("--top-k", type=int, default=15, help="检索返回的文档片段数量")

    # LLM 参数
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--max-completion-tokens", type=int, default=1200, help="最大生成 token 数")

    # 交互 / 单次提问
    parser.add_argument("--question", type=str, default=None, help="单次提问；如果不传则进入交互模式")
    parser.add_argument("--interactive-config", action="store_true", help="启动时主动询问缺失配置")

    # 网络选项
    parser.add_argument("--no-proxy", type=str, default=None, help="自定义 NO_PROXY，例如 aliyuncs.com,dashscope.aliyuncs.com")

    return parser.parse_args()


def prompt_if_missing(value: Optional[str], prompt_text: str, secret: bool = False) -> str:
    """
    当配置项缺失时进行交互式输入。
    对于 API_KEY 这类敏感信息，支持隐藏输入内容。
    """
    if value is not None and str(value).strip() != "":
        return str(value).strip()

    if secret:
        return getpass.getpass(prompt_text).strip()
    return input(prompt_text).strip()


def resolve_config(args: argparse.Namespace) -> AppConfig:
    """
    综合命令行参数与环境变量，生成最终配置对象。
    这里也会负责检查 Zotero 路径是否存在，并创建工作目录。
    """
    zotero_path = args.zotero_path or os.getenv("ASKMYZOTERO_ZOTERO_PATH")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    no_proxy = args.no_proxy or os.getenv("NO_PROXY")

    if args.interactive_config or not zotero_path:
        zotero_path = prompt_if_missing(zotero_path, "请输入 Zotero 本地 storage 路径: ")

    if args.interactive_config or not api_key:
        api_key = prompt_if_missing(
            api_key,
            "请输入 API_KEY（若你的本地兼容服务不校验，也建议填一个占位值如 EMPTY）: ",
            secret=True,
        )

    if args.interactive_config and base_url is None:
        base_url = input("请输入 BASE_URL（OpenAI 官方可直接回车留空）: ").strip() or None

    zotero_path_obj = Path(zotero_path).expanduser().resolve()
    if not zotero_path_obj.exists():
        raise FileNotFoundError(f"Zotero 路径不存在: {zotero_path_obj}")

    if not zotero_path_obj.is_dir():
        raise NotADirectoryError(f"Zotero 路径不是文件夹: {zotero_path_obj}")

    work_dir = Path(args.work_dir).expanduser().resolve()
    index_root = work_dir / args.index_name
    db_save_path = index_root / "faiss_index"
    splits_cache_path = index_root / "zotero_splits_cache.pkl"
    manifest_path = index_root / "manifest.json"

    index_root.mkdir(parents=True, exist_ok=True)

    if no_proxy:
        os.environ["NO_PROXY"] = no_proxy

    return AppConfig(
        zotero_path=zotero_path_obj,
        work_dir=work_dir,
        index_name=args.index_name,
        db_save_path=db_save_path,
        splits_cache_path=splits_cache_path,
        manifest_path=manifest_path,
        api_key=api_key,
        base_url=base_url,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        rebuild=args.rebuild,
        question=args.question,
        no_proxy=no_proxy,
    )


def print_config_summary(config: AppConfig) -> None:
    """
    打印当前运行配置摘要。
    这样可以在程序启动时快速确认路径、模型和索引目录是否正确。
    """
    masked_key = config.api_key[:6] + "..." if config.api_key else "(空)"
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"Zotero 路径        : {config.zotero_path}")
    print(f"API_KEY            : {masked_key}")
    print(f"BASE_URL           : {config.base_url or '(默认官方 OpenAI)'}")
    print(f"聊天模型           : {config.chat_model}")
    print(f"嵌入模型           : {config.embedding_model}")
    print(f"索引目录           : {config.db_save_path}")
    print(f"切块缓存           : {config.splits_cache_path}")
    print(f"Manifest 路径      : {config.manifest_path}")
    print(f"chunk_size         : {config.chunk_size}")
    print(f"chunk_overlap      : {config.chunk_overlap}")
    print(f"top_k              : {config.top_k}")
    print(f"temperature        : {config.temperature}")
    print(f"max_completion_tok : {config.max_completion_tokens}")
    print("=" * 60 + "\n")