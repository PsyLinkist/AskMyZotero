# 说明：
# 本文件负责统一管理命令行参数、环境变量读取和运行配置对象。
# 其他模块只依赖 AppConfig，不再各自维护零散的路径、模型和 API 参数。

import os
import getpass
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

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


#====================修改resolve_config函数
# region 原来resolve_config函数

# endregion
def resolve_config(args: argparse.Namespace) -> AppConfig:
    """
    综合命令行、环境变量与 YAML 文件，生成最终的 AppConfig 对象。
    """
    # 1. 优先尝试读取 config.yaml
    yaml_config = {}
    config_file = Path("config.yaml")
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
            print(f"📖 已从 {config_file} 加载配置")
        except Exception as e:
            print(f"⚠️ 读取 config.yaml 出错: {e}")

    # 2. 确定基础路径与 API Key (优先级: 命令行 > 环境变量 > YAML)
    zotero_path = args.zotero_path or os.getenv("ASKMYZOTERO_ZOTERO_PATH") or yaml_config.get("zotero_path")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or yaml_config.get("api_key")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or yaml_config.get("base_url")

    # 3. 交互式补充缺失的必填项 (防止程序直接崩溃)
    if not zotero_path:
        zotero_path = prompt_if_missing(None, "请输入 Zotero 本地 storage 路径: ")
    
    if not api_key:
        api_key = prompt_if_missing(None, "请输入 API_KEY: ", secret=True)

    # 4. 路径对象转换与工作目录创建
    zotero_path_obj = Path(zotero_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    index_root = work_dir / args.index_name
    index_root.mkdir(parents=True, exist_ok=True)

    # 5. 按照 AppConfig 的结构准备所有 18 个字段
    # 这样写的好处是：一目了然，且不容易漏掉字段
    config_dict = {
        # 路径类
        "zotero_path": zotero_path_obj,
        "work_dir": work_dir,
        "index_name": args.index_name,
        "db_save_path": index_root / "faiss_index",
        "splits_cache_path": index_root / "zotero_splits_cache.pkl",
        "manifest_path": index_root / "manifest.json",

        # API 类
        "api_key": api_key,
        "base_url": base_url,

        # 模型类 (优先读 YAML)
        "chat_model": yaml_config.get("chat_model") or args.chat_model,
        "embedding_model": yaml_config.get("embedding_model") or args.embedding_model,

        # 算法参数类 (优先读 YAML)
        "chunk_size": yaml_config.get("chunk_size") or args.chunk_size,
        "chunk_overlap": yaml_config.get("chunk_overlap") or args.chunk_overlap,
        "top_k": yaml_config.get("top_k") or args.top_k,

        # LLM 参数 (优先读 YAML)
        "temperature": yaml_config.get("temperature") or args.temperature,
        "max_completion_tokens": yaml_config.get("max_completion_tokens") or args.max_completion_tokens,

        # 运行状态类
        "rebuild": args.rebuild if args.rebuild else yaml_config.get("rebuild", False),
        "question": args.question,
        "no_proxy": args.no_proxy or os.getenv("NO_PROXY")
    }

    # 6. 使用 ** 技巧，把字典里的所有内容“解包”给 AppConfig
    # 这等同于 return AppConfig(zotero_path=..., api_key=..., ...)
    return AppConfig(**config_dict)



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