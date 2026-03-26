import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
# === 适配新版 LangChain 的核心组件 ===
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pickle

from tqdm import tqdm
# ==========================================
# 1. 配置参数 (请替换为你自己的信息)
# ==========================================
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 替换为你的千问 API Key
# 新增：强制让 python 访问阿里云时不走系统代理/VPN
os.environ["NO_PROXY"] = "aliyuncs.com,dashscope.aliyuncs.com"
ZOTERO_PATH = r"D:\ZoteroData\storage" 
DB_SAVE_PATH = "zotero_faiss_index"
SPLITS_CACHE_PATH = "zotero_splits_cache.pkl"  # 新增：文本块缓存文件的路径
# ==========================================
# 2. 构建或加载向量知识库 (带中间状态缓存版)
# ==========================================
def get_vectorstore():
    # 第一关：检查是否已经有最终的向量数据库
    if os.path.exists(DB_SAVE_PATH):
        print("🟢 正在加载已存在的文献向量数据库...")
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        vectorstore = FAISS.load_local(DB_SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # 第二关：检查是否有切好的文本块缓存
    if os.path.exists(SPLITS_CACHE_PATH):
        print(f"🟢 发现文本块缓存文件，跳过 PDF 解析，直接读取...")
        with open(SPLITS_CACHE_PATH, "rb") as f:
            splits = pickle.load(f)
        print(f"✅ 成功从缓存中加载了 {len(splits)} 个文本块！")
    else:
        # 如果什么都没有，才老老实实去读 PDF
        print(f"🟡 未发现数据库和缓存，准备扫描目录: {ZOTERO_PATH}")
        print("⏳ 正在读取 PDF 文件，请观察下方的进度条...")
        loader = DirectoryLoader(
            ZOTERO_PATH, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader, 
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()
        print(f"\n✅ PDF 读取完毕！共成功提取了 {len(docs)} 页文献内容。")

        print("✂️ 正在切割文献文本以适应 AI 的胃口...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        print(f"✅ 文本切割完成！共生成了 {len(splits)} 个文本块。")
        
        # 【关键新增】将切好的文本块保存到本地硬盘
        print("💾 正在将切割好的文本块保存到本地缓存文件...")
        with open(SPLITS_CACHE_PATH, "wb") as f:
            pickle.dump(splits, f)
        print("✅ 文本块缓存保存成功！以后重试就不用再解析 PDF 啦。")

    # --- 下面是向量化和上传的逻辑 ---
    print("☁️ 正在调用千问 API 将文本块转化为向量并存入数据库...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    
    batch_size = 300
    vectorstore = None
    
    for i in tqdm(range(0, len(splits), batch_size), desc="🚀 向量化上传进度"):
        batch = splits[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
        else:
            vectorstore.add_documents(batch)
            
    vectorstore.save_local(DB_SAVE_PATH)
    print("🎉 数据库构建并保存完毕！以后启动就是秒开了。")
    return vectorstore

# ==========================================
# 3. 初始化 RAG 对话链 (加入中间状态拦截日志)
# ==========================================
def create_chat_chain(vectorstore):
    # 请确保这里的模型是你目前有额度、且支持文本的，比如 qwen-max
    llm = ChatTongyi(model="qwen-max") 
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