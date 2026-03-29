import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# 引入阿里原生组件以进行回退处理和支持
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# === 适配新版 LangChain 的核心组件 ===
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pickle

from tqdm import tqdm
# ==========================================
# 1. 配置参数 (默认值)
# ==========================================
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 替换为你的千问 API Key
os.environ["NO_PROXY"] = "aliyuncs.com,dashscope.aliyuncs.com"
ZOTERO_PATH = r"XXXXXXXXXXXXX"  # 在界面上填
DB_SAVE_PATH = "zotero_faiss_index"
SPLITS_CACHE_PATH = "zotero_splits_cache.pkl"

# ==========================================
# 2. 构建或加载向量知识库 (带中间状态缓存版)
# ==========================================
def get_vectorstore(zotero_path=None, api_key=None, base_url=None, embedding_model="text-embedding-v3", status_callback=print, progress_callback=None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not zotero_path:
        zotero_path = ZOTERO_PATH

    # 智能分发逻辑：根据用户填写的 Base URL 选择最优的基础组件
    if base_url and "dashscope.aliyuncs.com" in base_url:
        status_callback("🟢 检测到使用阿里云节点，切换至原生 DashScopeEmbeddings 以避免兼容层报错...")
        embeddings = DashScopeEmbeddings(dashscope_api_key=api_key, model=embedding_model)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=base_url, model=embedding_model)

    # 第一关：检查是否已经有最终的向量数据库
    if os.path.exists(DB_SAVE_PATH):
        status_callback("🟢 正在加载已存在的文献向量数据库...")
        vectorstore = FAISS.load_local(DB_SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # 第二关：检查是否有切好的文本块缓存
    if os.path.exists(SPLITS_CACHE_PATH):
        status_callback(f"🟢 发现文本块缓存文件，跳过 PDF 解析，直接读取...")
        with open(SPLITS_CACHE_PATH, "rb") as f:
            splits = pickle.load(f)
        status_callback(f"✅ 成功从缓存中加载了 {len(splits)} 个文本块！")
    else:
        # 如果什么都没有，才老老实实去读 PDF
        from pathlib import Path
        pdf_files = list(Path(zotero_path).rglob("*.pdf"))
        total_files = len(pdf_files)
        
        if total_files == 0:
            status_callback(f"⚠️ 在 {zotero_path} 下没有找到任何 PDF 文件。")
            return None
            
        status_callback(f"🟡 未发现数据库和缓存，扫描目录发现 {total_files} 个 PDF 文件，准备解析...")
        docs = []
        for i, pdf_file in enumerate(pdf_files):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs.extend(loader.load())
            except Exception as e:
                pass # 忽略损坏的PDF
            if progress_callback:
                # 前 50% 进度用于读取 PDF
                progress = (i + 1) / total_files * 0.5
                progress_callback(progress, f"正在读取 PDF文献 ({i+1}/{total_files})...已提取 {len(docs)} 页")
                
        status_callback(f"✅ PDF 读取完毕！共成功提取了 {len(docs)} 页文献内容。")

        status_callback("✂️ 正在切割文献文本以适应 AI 的胃口...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        status_callback(f"✅ 文本切割完成！共生成了 {len(splits)} 个文本块。")
        
        # 【关键新增】将切好的文本块保存到本地硬盘
        status_callback("💾 正在将切割好的文本块保存到本地缓存文件...")
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(splits, f)
        status_callback("✅ 文本块缓存保存成功！以后重试就不用再解析 PDF 啦。")

    # --- 下面是向量化和上传的逻辑 ---
    status_callback(f"☁️ 正在调用 API 将文本块转化为向量({embedding_model})并存入数据库...")
    
    batch_size = 300
    vectorstore = None
    
    total_splits = len(splits)
    for i in range(0, total_splits, batch_size):
        batch = splits[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
        else:
            vectorstore.add_documents(batch)
            
        if progress_callback:
            # 后 50% 进度用于向API上传 embedding
            # 判断是在解析PDF后调用，还是读取缓存后直接调用的。
            # 为了平滑，如果是缓存加载直接占100%区间，不过就用参数传文字吧
            uploaded_count = min(i + batch_size, total_splits)
            base_progress = 0.5 if 'Docs' in locals() or 'docs' in locals() else 0.0
            scale = 0.5 if 'Docs' in locals() or 'docs' in locals() else 1.0
            
            current_progress = base_progress + (uploaded_count / total_splits) * scale
            progress_callback(current_progress, f"向量化并上传进度: {uploaded_count}/{total_splits} 个文本块")
            
    vectorstore.save_local(DB_SAVE_PATH)
    status_callback("🎉 数据库构建并保存完毕！以后启动就是秒开了。")
    return vectorstore

# ==========================================
# 3. 初始化 RAG 对话链 (加入中间状态拦截日志)
# ==========================================
def create_chat_chain(vectorstore, api_key=None, base_url=None, chat_model="qwen-max"):
    # 为了解决千问直接接入 OpenAI 兼容层时偶发的 input.contents 报 400 格式错误
    # 我们做一次优雅的智能降级分发
    if base_url and "dashscope.aliyuncs.com" in base_url:
        llm = ChatTongyi(dashscope_api_key=api_key, model=chat_model)
    else:
        llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=chat_model) 
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # 只把最相关的前 4 个文本块提取出来

    system_prompt = (
        "你是一个严谨的学术科研助手。"
        "请务必【仅根据以下检索到的文献片段】来回答用户的问题。"
        "如果在文献片段中找不到答案，请诚实地回答“根据当前文献库无法回答该问题”，绝不能自己编造数据或结论。"
        "\n\n"
        "检索到的文献片段：\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 【巧妙的改动】在组装文档的这个环节，顺手打印一条日志！
    def format_docs(docs):
        print(f"\n📚 [底层日志] 知识库检索完成！共匹配到 {len(docs)} 个相关的文献片段。")
        print("🧠 [底层日志] 正在将片段投喂给千问大模型进行深度阅读和总结，请稍候...\n")
        return "\n\n".join(doc.page_content for doc in docs)

    # 组装最新版的 LCEL 链
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ==========================================
# 4. 启动命令行对话交互 (加入流式打字机效果)
# ==========================================
if __name__ == "__main__":
    vectorstore = get_vectorstore()
    rag_chain = create_chat_chain(vectorstore)

    print("\n" + "="*50)
    print("🤖 Zotero 文献库 AI 助手已启动！(输入 'quit' 退出)")
    print("="*50 + "\n")

    while True:
        user_input = input("🧑‍🎓 你: ")
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！祝你科研顺利。")
            break
            
        print("🔍 AI 正在把你的问题转换为向量，并去知识库中翻找...")
        
        print("💡 回答: ", end="", flush=True) # 准备输出回答
        
        try:
            # 【关键修改】把 invoke 换成 stream，实现逐字打印！
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n❌ 对话时发生网络或额度错误: {e}")
            
        print("\n" + "-" * 50)