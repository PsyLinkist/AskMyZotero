import streamlit as st
import os
import re
from zotero_rag import get_vectorstore, create_chat_chain, DB_SAVE_PATH

def get_default_zotero_path():
    # 尝试从配置文件中读取自定义路径
    appdata = os.environ.get('APPDATA')
    if appdata:
        profiles_dir = os.path.join(appdata, "Zotero", "Zotero", "Profiles")
        if os.path.exists(profiles_dir):
            for profile in os.listdir(profiles_dir):
                prefs_path = os.path.join(profiles_dir, profile, "prefs.js")
                if os.path.exists(prefs_path):
                    try:
                        with open(prefs_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            match = re.search(r'user_pref\("extensions\.zotero\.dataDir",\s*"(.*?)"\);', content)
                            if match:
                                # 处理转义的路径分隔符
                                path = match.group(1).replace('\\\\', '\\')
                                print(f"✅ 自动检测到 Zotero 配置文件中的路径: {path}")
                                if os.path.exists(path):
                                    return path
                    except Exception:
                        pass
    
    # 默认路径
    default_path = os.path.expanduser(r"~\Zotero")
    if os.path.exists(default_path):
        print(f"✅ 自动检测到默认 Zotero 路径: {default_path}")
        return default_path
        
    return None

st.set_page_config(page_title="AskMyZotero", page_icon="📚", layout="wide")

# =============== 侧边栏配置区 ===============
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 获取默认或者现有的环境变量供用户参考
    default_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    api_key_input = st.text_input("API Key", value=default_api_key, type="password", help="在此输入你的 API 密钥")
    base_url_input = st.text_input("Base URL", value="https://api.key77qiqi.cn/v1", help="支持千问、DeepSeek、本地vLLM等的兼容URL")
    
    auto_path = get_default_zotero_path()
    if auto_path:
        st.success(f"✅ 已自动关联 Zotero 路径:\n`{auto_path}`")
        zotero_path_input = auto_path
    else:
        st.warning("⚠️ 未能自动找到 Zotero 数据路径，请手动输入！")
        zotero_path_input = st.text_input("Zotero 文件夹路径", help="您的 Zotero 附件所在的绝对路径")
    
    embedding_model_input = st.text_input("向量化模型", value="text-embedding-3-small", help="例如: text-embedding-v3 等 DashScope 支持的模型")
    chat_model_input = st.text_input("对话模型", value="gpt-4o-mini", help="例如: qwen-max, qwen-turbo 等")
    
    st.divider()
    
    # 状态检查
    db_exists = os.path.exists(DB_SAVE_PATH)
    if db_exists:
        st.success("✅ 检测到本地已存在向量数据库。")
    else:
        st.warning("⚠️ 本地未检测到向量数据库，请点击下方进行初始化。")

    init_btn = st.button("🚀 初始化 / 加载知识库", type="primary")

# 定义加载过程，不再使用 st.cache_resource 强行缓存带有状态界面的函数，
# 而是拆分开，将耗时的操作提取出来，然后把返回的链存入 session_state

def init_knowledge_base(zotero_path, api_key, base_url, embedding_model, chat_model):
    # 定义 Streamlit 状态组件占位
    status_container = st.empty()
    progress_bar = st.progress(0, text="正在扫描文献库...")
    
    def status_callback(msg):
        status_container.info(msg)
        
    def progress_callback(progress, text):
        # 确保进度在 0.0 到 1.0 之间
        progress = max(0.0, min(1.0, progress))
        progress_bar.progress(progress, text=text)

    # 调用核心方法获取向量库
    vectorstore = get_vectorstore(
        zotero_path=zotero_path, 
        api_key=api_key, 
        base_url=base_url,
        embedding_model=embedding_model,
        status_callback=status_callback, 
        progress_callback=progress_callback
    )
    
    # 获取完后修改界面提示
    if vectorstore:
        progress_bar.empty()
        status_container.success("🎉 知识库准备就绪，可以开始提问啦！")
        return create_chat_chain(vectorstore, api_key=api_key, base_url=base_url, chat_model=chat_model)
    else:
        status_container.error("❌ 知识库初始化失败，请检查路径。")
        return None

st.title("📚 AskMyZotero - 文献库 AI 助手")

# 如果用户点击初始化，则运行构建逻辑
if init_btn:
    if not api_key_input:
        st.error("请在左侧配置栏先填写 API Key！")
        st.stop()
    if not zotero_path_input:
        st.error("未找到 Zotero 数据路径，请在左侧侧边栏手动填写！")
        st.stop()
        
    try:
        # 重建知识库并存入 session state
        st.session_state.rag_chain = init_knowledge_base(zotero_path_input, api_key_input, base_url_input, embedding_model_input, chat_model_input)
    except Exception as e:
        st.error(f"初始化或加载模型知识库失败: {e}")
        st.stop()
        
# 如果没点击按钮但本地有数据库且还没加载，并且用户主动来回车聊天，静默加载即可
elif "rag_chain" not in st.session_state and db_exists:
    with st.spinner("检测到本地已有知识库，正在极速加载..."):
        st.session_state.rag_chain = init_knowledge_base(zotero_path_input, api_key_input, base_url_input, embedding_model_input, chat_model_input)

# =============== 以下是聊天交互区 ===============
if "rag_chain" in st.session_state and st.session_state.rag_chain is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是你的 Zotero 文献库助理。你的文献档案已加载完成，请问有什么可以帮到你？"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您关于文献的问题（支持流式输出）..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.rag_chain.stream(prompt):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"对话发生错误: {e}")
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👈 请在左侧侧边栏配置您的 API Key 和 Zotero 路径，然后点击**初始化/加载知识库**按钮开始使用！")
