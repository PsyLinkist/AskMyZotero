"""本文件负责构建、加载和使用向量索引，以及组装 RAG 问答链。"""

import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import AppConfig
from src.metadata_store import rebuild_metadata_store
from src.parser import load_or_create_splits


def build_embeddings(config: AppConfig) -> OpenAIEmbeddings:
    """根据配置构建嵌入模型客户端。"""
    kwargs: dict[str, Any] = {
        "model": config.embedding_model,
        "api_key": config.api_key,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAIEmbeddings(**kwargs)


def build_llm(config: AppConfig) -> ChatOpenAI:
    """根据配置构建聊天模型客户端。"""
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


def remove_path_if_exists(path: Path) -> None:
    """如果目标路径存在，则删除文件或目录。"""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_rebuild_if_needed(config: AppConfig) -> None:
    """在启用 --rebuild 时清理旧索引和缓存。"""
    if config.rebuild:
        print("🗑️ 检测到 --rebuild，正在删除旧索引与旧缓存...")
        remove_path_if_exists(config.db_save_path)
        remove_path_if_exists(config.splits_cache_path)
        remove_path_if_exists(config.metadata_db_path)
        print("✅ 旧索引与旧缓存已删除。")

# yzx update for PO
# def build_vectorstore_from_splits(config: AppConfig, splits) -> FAISS:
#     """根据文本块构建并保存 FAISS 向量库。"""
#     print("☁️ 正在调用嵌入模型生成向量并构建 FAISS 索引...")
#     embeddings = build_embeddings(config)

#     batch_size = 64
#     vectorstore = None

#     for i in tqdm(range(0, len(splits), batch_size), desc="🚀 向量化进度"):
#         batch = splits[i: i + batch_size]
#         print(
#             f"   … 嵌入批次 {i // batch_size + 1}，本批 {len(batch)} 条（等待 API 返回中，无反应时请检查网络与 base_url）",
#             flush=True,
#         )
#         if vectorstore is None:
#             vectorstore = FAISS.from_documents(batch, embeddings)
#         else:
#             vectorstore.add_documents(batch)

#     if vectorstore is None:
#         raise RuntimeError("向量库构建失败，未生成任何向量。")

#     remove_path_if_exists(config.db_save_path)
#     vectorstore.save_local(str(config.db_save_path))
#     print(f"🎉 向量数据库构建并保存完成: {config.db_save_path}")
#     return vectorstore

def build_vectorstore_from_splits(config: AppConfig, splits) -> FAISS:
    """根据文本块构建并保存 FAISS 向量库。"""
    print("☁️ 正在调用嵌入模型生成向量并构建 FAISS 索引...")
    embeddings = build_embeddings(config)

    batch_size = 64
    vectorstore = None
    
    # [新增] 计算总批次，用于计算进度百分比
    total_batches = (len(splits) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(splits), batch_size), desc="🚀 向量化进度"):
        batch = splits[i: i + batch_size]
        current_batch = i // batch_size + 1
        
        # --- [新增] 触发进度回调 ---
        # 安全地从 config 中尝试获取进度回调函数
        callback = getattr(config, 'progress_callback', None)
        if callback:
            callback(
                stage="embedding",
                current=current_batch,
                total=total_batches,
                message=f"正在生成向量 (批次 {current_batch}/{total_batches})"
            )
        # --------------------------

        print(
            f"   … 嵌入批次 {current_batch}，本批 {len(batch)} 条（等待 API 返回中，无反应时请检查网络与 base_url）",
            flush=True,
        )
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        raise RuntimeError("向量库构建失败，未生成任何向量。")

    remove_path_if_exists(config.db_save_path)
    vectorstore.save_local(str(config.db_save_path))
    print(f"🎉 向量数据库构建并保存完成: {config.db_save_path}")
    return vectorstore

def get_vectorstore(config: AppConfig) -> FAISS:
    """优先加载本地索引，不存在时重新构建。"""
    ensure_rebuild_if_needed(config)
    embeddings = build_embeddings(config)
    splits = load_or_create_splits(config)

    if config.db_save_path.exists():
        print(f"🟢 正在加载已存在的向量数据库: {config.db_save_path}")
        vectorstore = FAISS.load_local(
            str(config.db_save_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = build_vectorstore_from_splits(config, splits)

    rebuild_metadata_store(config.metadata_db_path, splits, vectorstore=vectorstore)
    return vectorstore


def create_chat_chain(config: AppConfig, vectorstore: FAISS):
    """基于向量库构建 RAG 对话链。"""
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
        [("system", system_prompt), ("human", "{input}")]
    )

    def format_docs(docs) -> str:
        """将检索结果整理成可读上下文字符串。"""
        print(f"\n📚 [底层日志] 检索完成，共命中 {len(docs)} 个相关片段。")
        print("🧠 [底层日志] 正在组织上下文并调用大模型生成回答...\n")

        formatted_parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)

            header = f"[片段 {idx}] source={source}"
            if isinstance(page, int):
                header += f", page={page + 1}"
            formatted_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(formatted_parts)

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_once(rag_chain, question: str) -> None:
    """对单个问题执行一次流式问答。"""
    print("🔍 AI 正在检索相关文献片段...")
    print("💡 回答: ", end="", flush=True)

    try:
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n❌ 对话时发生错误: {e}")

    print("\n" + "-" * 60)


def interactive_chat(rag_chain) -> None:
    """启动命令行交互式问答循环。"""
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


def api_ask_once(rag_chain, question: str) -> dict:
    """供 API 调用的一次性问答接口。"""
    try:
        answer_text = rag_chain.invoke(question)
        return {"success": True, "answer": answer_text}
    except Exception as e:
        return {"success": False, "answer": "", "error_message": str(e)}
