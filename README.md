# 📚 Zotero 本地文献库 AI 对话助手 (Zotero RAG Assistant)

本项目是一个基于 Python 和 LangChain 构建的本地文献检索增强生成（RAG）系统。它可以将你 Zotero 中的 PDF 文献转化为本地向量知识库，并通过对接阿里云通义千问大模型（Qwen），让你能够以自然语言与你的个人文献库进行对话。

## ✨ 功能特性

* **📂 自动解析 Zotero 目录**：穿透 Zotero 复杂的底层文件结构，自动提取所有 PDF。
* **💾 双重本地缓存**：支持文本块（pkl）和向量库（FAISS）本地保存，首次构建后，后续启动只需 **1秒**。
* **🚀 极速断点续传**：带有 `tqdm` 进度条，支持分批次向量化上传，防止大批量文献导致网络超时。
* **🛡️ 代理免疫**：内置环境变量脱敏，开着科学上网工具（梯子）依然能稳定连接国内 API。
* **⌨️ 流式打字机输出**：对话时无需漫长等待，AI 思考过程实时逐字展现。

---

## 🛠️ 环境准备与依赖安装

请确保你的电脑已安装 Python（建议 3.8 - 3.11 版本）。打开终端，运行以下命令安装所需的核心依赖库：

```bash
python -m pip install langchain langchain-community langchain-core dashscope faiss-cpu pypdf tqdm
```

---

## ⚙️ 核心配置指南

在运行代码前，请务必打开 `zotero_rag.py`，根据你个人的情况修改以下配置：

### 1. 配置通义千问 API Key

在阿里云百炼控制台左侧“API Key”那一栏，申请你的 API Key，并替换代码中的对应位置。

> **⚠️ 警告**：不要将带有真实 API Key 的代码上传到公开的 GitHub 仓库，以免额度被盗刷！

```python
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
```

### 2. 配置 Zotero 本地路径

将 `ZOTERO_PATH` 修改为你电脑上 Zotero 真实存放 PDF 附件的 `storage` 文件夹路径：

* **Windows 示例**: `r"D:\ZoteroData\storage"` 或 `r"C:\Users\用户名\Zotero\storage"`
* **Mac 示例**: `"/Users/用户名/Zotero/storage"`

```python
ZOTERO_PATH = r"D:\ZoteroData\storage" 
```

### 3. 配置 AI 模型 (按需修改)

代码中使用了两个不同的模型，分别负责“阅读”和“回答”：

* **Embedding 模型（文本转向量）**：在 `get_vectorstore()` 函数中，默认配置为 `text-embedding-v3`。如果你的该模型免费额度耗尽，请确保你的阿里云账户有余额并关闭了“用完即停”, **建议先充值1块钱，如果你的本地论文特别多的话，直接在头像处的费用与成本点击充值1块**。
* **LLM 对话模型**：在 `create_chat_chain()` 函数中，默认配置为旗舰模型 `qwen-max`。你可以根据成本和需求将其修改为 `qwen-turbo` 或 `qwen-plus`。

---

## 🚀 运行说明

在终端中执行以下命令启动助手：

```bash
python -u "zotero_rag.py"
```

* **首次运行**：程序会扫描读取所有的 PDF -> 切割文本 -> 保存为 `pkl` 缓存 -> 批量上传至阿里云进行向量化 -> 保存为本地 `FAISS` 数据库。此过程可能需要几分钟到十几分钟（取决于文献数量）。
* **非首次运行**：程序会瞬间检测到本地的 `zotero_faiss_index` 文件夹，1秒钟直接启动对话交互界面。

---

## 🔧 进阶调优与常见问题

### 1. 如何调整 AI 阅读的文献片段数量？

在 `create_chat_chain()` 函数中，有一个参数为 `k=4`：

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

这代表每次提问 AI 会阅读最相关的 4 个文本块。如果你觉得 AI 经常漏掉信息，可以改为 `6` 或 `8`；如果追求极致速度和省钱，可以改为 `2`。

### 2. 报错 `url error` 或 `InvalidParameter`？

通常是因为配置了错误的大模型名称（如使用了多模态大模型 `qvq` 等不兼容纯文本链的模型）。请确保 `llm = ChatTongyi(model="qwen-max")` 填写的是官方支持的文本模型。

### 3. 更换 Embedding 模型后报错？

如果你把代码里的 `text-embedding-v3` 换成了 `v4` 或其他模型，**原有的向量库将完全失效**。
**解决办法**：请手动删除项目文件夹下的 `zotero_faiss_index` 文件夹和 `zotero_splits_cache.pkl` 文件，然后重新运行代码，让其重新构建匹配新模型的向量数据库。

---

**这份 README 已经为你梳理好了所有的核心逻辑和避坑指南。你要不要试着把它存下来，然后用我们最终版的代码去向你的文献库提第一个真实的学术问题试试看？**
