# 说明：
# 本文件负责向量库和问答链的核心逻辑，包括 embeddings、LLM、FAISS 构建/加载以及 RAG 对话链。
# 它相当于系统的“索引层 + 检索问答层”，主程序只需要调用这里的高层函数即可。

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
from src.parser import load_or_create_splits


def build_embeddings(config: AppConfig) -> OpenAIEmbeddings:
    """
    根据配置构建 OpenAI 风格的嵌入模型客户端。
    这里统一封装 api_key、base_url 和 embedding_model，避免在别处重复写。
    """
    kwargs: dict[str, Any] = {
        "model": config.embedding_model,
        "api_key": config.api_key,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAIEmbeddings(**kwargs)


def build_llm(config: AppConfig) -> ChatOpenAI:
    """
    根据配置构建 OpenAI 风格的聊天模型客户端。
    这里统一处理模型名、温度、最大 token 数和重试参数。
    """
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
    """
    如果目标路径存在，则将其删除。
    这个工具函数会同时处理文件和文件夹两种情况。
    """
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_rebuild_if_needed(config: AppConfig) -> None:
    """
    当用户传入 --rebuild 时，删除旧索引和旧切块缓存。
    这样程序下一步就会走完整的重新解析与重新建库流程。
    """
    if config.rebuild:
        print("🗑️ 检测到 --rebuild，正在删除旧索引与旧缓存...")
        remove_path_if_exists(config.db_save_path)
        remove_path_if_exists(config.splits_cache_path)
        print("✅ 旧索引与旧缓存已删除。")


def build_vectorstore_from_splits(config: AppConfig, splits) -> FAISS:
    """
    根据文本块列表构建 FAISS 向量库，并将结果保存到本地。
    这里采用分批写入的方式，以降低大规模文献向量化时的内存压力。
    """
    print("☁️ 正在调用嵌入模型生成向量并构建 FAISS 索引...")
    embeddings = build_embeddings(config)

    # 较小批次：tqdm 步进更勤、单次 HTTP 压力更小，第三方网关也不容易长时间无响应
    batch_size = 64
    vectorstore = None

    for i in tqdm(range(0, len(splits), batch_size), desc="🚀 向量化进度"):
        batch = splits[i: i + batch_size]
        print(
            f"   … 嵌入批次 {i // batch_size + 1}，本批 {len(batch)} 条（等待 API 返回中，无反应时请检查网络与 base_url）",
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
    """
    优先加载已有 FAISS 索引；如果本地索引不存在，则从文本块开始构建新索引。
    这个函数对上层屏蔽了“加载还是构建”的具体细节。
    """
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


def create_chat_chain(config: AppConfig, vectorstore: FAISS):
    """
    基于向量库构建 RAG 对话链。
    检索器会先找出相关片段，再将上下文与用户问题一并交给聊天模型生成答案。
    """
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
        """
        将检索到的文档片段整理为可读上下文字符串。
        同时打印命中数量和来源信息，方便调试检索效果。
        """
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

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def answer_once(rag_chain, question: str) -> None:
    """
    对单个问题执行一次检索问答，并以流式方式输出答案。
    这适合命令行单问单答模式，也会被交互模式重复调用。
    """
    print("🔍 AI 正在检索相关文献片段...")
    print("💡 回答: ", end="", flush=True)

    try:
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n❌ 对话时发生错误: {e}")

    print("\n" + "-" * 60)


def interactive_chat(rag_chain) -> None:
    """
    启动命令行交互式问答循环。
    用户可以持续提问，直到输入 quit、exit 或 退出。
    """
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



# 补充接口（lx）

def api_ask_once(rag_chain, question: str) -> dict:
    """
    【API 专用接口】
    接收问题，调用 RAG 链并返回完整字符串，供 Agent/FastAPI 组装返回给前端。
    不使用流式 print 输出。
    """
    try:
        # invoke 会直接拿到完整的回答，而不是一点点打印
        answer_text = rag_chain.invoke(question)
        
        return {
            "success": True,
            "answer": answer_text
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "error_message": str(e)
        }