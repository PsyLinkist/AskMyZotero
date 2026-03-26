# 说明：
# 本代码用于将 Zotero 本地文献库构建为 FAISS 向量库，并通过 OpenAI 风格接口进行 RAG 问答。
# 支持命令行输入 Zotero 路径、API_KEY、BASE_URL、聊天模型、嵌入模型等参数。
# 如果目标索引目录或缓存文件已存在，只有在 --rebuild 时才会先删除旧文件并重建。

import os
import shutil
import pickle
import getpass
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. 配置对象
# ==========================================
@dataclass
class AppConfig:
    zotero_path: Path
    work_dir: Path
    index_name: str
    db_save_path: Path
    splits_cache_path: Path

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


# ==========================================
# 2. 参数解析
# ==========================================
def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--top-k", type=int, default=4, help="检索返回的文档片段数量")

    # LLM 参数
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--max-completion-tokens", type=int, default=1200, help="最大生成 token 数")

    # 交互 / 单次提问
    parser.add_argument("--question", type=str, default=None, help="单次提问；如果不传则进入交互模式")
    parser.add_argument("--interactive-config", action="store_true", help="启动时主动询问缺失配置")

    # 网络选项
    parser.add_argument("--no-proxy", type=str, default=None, help="自定义 NO_PROXY，例如 aliyuncs.com,dashscope.aliyuncs.com")

    return parser.parse_args()


# ==========================================
# 3. 配置收集与校验
# ==========================================
def prompt_if_missing(value: Optional[str], prompt_text: str, secret: bool = False) -> str:
    if value is not None and str(value).strip() != "":
        return str(value).strip()

    if secret:
        return getpass.getpass(prompt_text).strip()
    return input(prompt_text).strip()


def resolve_config(args: argparse.Namespace) -> AppConfig:
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

    index_root.mkdir(parents=True, exist_ok=True)

    if no_proxy:
        os.environ["NO_PROXY"] = no_proxy

    return AppConfig(
        zotero_path=zotero_path_obj,
        work_dir=work_dir,
        index_name=args.index_name,
        db_save_path=db_save_path,
        splits_cache_path=splits_cache_path,
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
    print(f"chunk_size         : {config.chunk_size}")
    print(f"chunk_overlap      : {config.chunk_overlap}")
    print(f"top_k              : {config.top_k}")
    print(f"temperature        : {config.temperature}")
    print(f"max_completion_tok : {config.max_completion_tokens}")
    print("=" * 60 + "\n")


# ==========================================
# 4. OpenAI 风格客户端构建
# ==========================================
def build_embeddings(config: AppConfig) -> OpenAIEmbeddings:
    kwargs = {
        "model": config.embedding_model,
        "api_key": config.api_key,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAIEmbeddings(**kwargs)


def build_llm(config: AppConfig) -> ChatOpenAI:
    kwargs = {
        "model": config.chat_model,
        "api_key": config.api_key,
        "temperature": config.temperature,
        "max_retries": 2,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = config.max_completion_tokens

    return ChatOpenAI(**kwargs)


# ==========================================
# 5. 文件系统辅助
# ==========================================
def remove_path_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_rebuild_if_needed(config: AppConfig) -> None:
    if config.rebuild:
        print("🗑️ 检测到 --rebuild，正在删除旧索引与旧缓存...")
        remove_path_if_exists(config.db_save_path)
        remove_path_if_exists(config.splits_cache_path)
        print("✅ 旧索引与旧缓存已删除。")


# ==========================================
# 6. 文档加载与切块
# ==========================================
def load_pdf_documents(zotero_path: Path):
    print(f"🟡 正在扫描目录: {zotero_path}")
    print("⏳ 正在读取 PDF 文件，请稍候...")

    loader = DirectoryLoader(
        str(zotero_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,
    )
    docs = loader.load()

    if not docs:
        raise RuntimeError("未读取到任何 PDF 内容，请检查 Zotero storage 路径是否正确。")

    print(f"✅ PDF 读取完毕！共提取了 {len(docs)} 页文献内容。")
    return docs


def split_documents(docs, chunk_size: int, chunk_overlap: int):
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


def load_or_create_splits(config: AppConfig):
    if config.splits_cache_path.exists():
        print(f"🟢 发现文本块缓存，直接读取: {config.splits_cache_path}")
        with open(config.splits_cache_path, "rb") as f:
            splits = pickle.load(f)
        print(f"✅ 缓存读取成功，共 {len(splits)} 个文本块。")
        return splits

    docs = load_pdf_documents(config.zotero_path)
    splits = split_documents(docs, config.chunk_size, config.chunk_overlap)

    print("💾 正在保存文本块缓存...")
    with open(config.splits_cache_path, "wb") as f:
        pickle.dump(splits, f)
    print("✅ 文本块缓存保存成功。")

    return splits


# ==========================================
# 7. 构建或加载向量库
# ==========================================
def build_vectorstore_from_splits(config: AppConfig, splits) -> FAISS:
    print("☁️ 正在调用嵌入模型生成向量并构建 FAISS 索引...")
    embeddings = build_embeddings(config)

    batch_size = 300
    vectorstore = None

    for i in tqdm(range(0, len(splits), batch_size), desc="🚀 向量化进度"):
        batch = splits[i: i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        raise RuntimeError("向量库构建失败，未生成任何向量。")

    if config.db_save_path.exists():
        shutil.rmtree(config.db_save_path)

    vectorstore.save_local(str(config.db_save_path))
    print(f"🎉 向量数据库构建并保存完成: {config.db_save_path}")
    return vectorstore


def get_vectorstore(config: AppConfig) -> FAISS:
    ensure_rebuild_if_needed(config)

    embeddings = build_embeddings(config)

    if config.db_save_path.exists():
        print(f"🟢 正在加载已存在的向量数据库: {config.db_save_path}")
        return FAISS.load_local(
            str(config.db_save_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    splits = load_or_create_splits(config)
    return build_vectorstore_from_splits(config, splits)


# ==========================================
# 8. 构建 RAG 对话链
# ==========================================
def create_chat_chain(config: AppConfig, vectorstore: FAISS):
    llm = build_llm(config)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.top_k})

    system_prompt = (
        "你是一个严谨的学术科研助手。"
        "请务必仅根据以下检索到的文献片段回答用户问题。"
        "如果文献片段中找不到答案，请明确回答："
        "“根据当前文献库无法回答该问题”。"
        "不要编造结论、数据或引用。"
        "\n\n"
        "检索到的文献片段如下：\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs) -> str:
        print(f"\n📚 [底层日志] 检索完成，共命中 {len(docs)} 个相关片段。")
        print("🧠 [底层日志] 正在组织上下文并调用大模型生成回答...\n")

        formatted_parts: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)

            header = f"[片段 {idx}] source={source}"
            if isinstance(page, int):
                header += f", page={page + 1}"

            formatted_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(formatted_parts)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ==========================================
# 9. 提问执行
# ==========================================
def answer_once(rag_chain, question: str) -> None:
    print("🔍 AI 正在检索相关文献片段...")
    print("💡 回答: ", end="", flush=True)

    try:
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n❌ 对话时发生错误: {e}")

    print("\n" + "-" * 60)


def interactive_chat(rag_chain) -> None:
    print("\n" + "=" * 60)
    print("🤖 Zotero 文献库 AI 助手已启动！(输入 quit / exit / 退出 结束)")
    print("=" * 60 + "\n")

    while True:
        user_input = input("🧑‍🎓 你: ").strip()
        if user_input.lower() in {"quit", "exit", "退出"}:
            print("再见！祝你科研顺利。")
            break

        if not user_input:
            continue

        answer_once(rag_chain, user_input)


# ==========================================
# 10. 程序入口
# ==========================================
def main():
    args = parse_args()
    config = resolve_config(args)
    print_config_summary(config)

    vectorstore = get_vectorstore(config)
    rag_chain = create_chat_chain(config, vectorstore)

    if config.question:
        answer_once(rag_chain, config.question)
    else:
        interactive_chat(rag_chain)


if __name__ == "__main__":
    main()