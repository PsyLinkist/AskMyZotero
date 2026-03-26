# 📚 AskMyZotero：Zotero 本地文献库 AI 对话助手

AskMyZotero 是一个基于 Python 与 LangChain 构建的本地文献检索增强生成（RAG）项目。  
它的目标是将 Zotero 本地文献库中的 PDF 文献解析、切块、向量化，并构建本地知识库，让用户能够通过自然语言与自己的个人文献库进行问答交互。

当前版本已经完成了**代码模块化拆分**，将原本的单文件脚本拆分为配置管理、文献扫描、文本解析、索引构建、快照管理与主入口等多个部分，便于后续继续优化 RAG 框架与实现增量同步。

---

## ✨ 功能特性

- **📂 自动扫描 Zotero 本地文献库**  
  自动遍历 Zotero `storage` 目录中的 PDF 文件，收集本地文献信息。

- **🧩 模块化代码结构**  
  将原始脚本拆分为 `config / scanner / parser / indexer / manifest / main` 六个主要模块，方便维护与扩展。

- **💾 本地缓存与索引保存**  
  支持文本块缓存与本地向量库存储，避免重复解析与重复向量化。

- **🔍 基础 RAG 问答能力**  
  将用户问题转为检索请求，从本地文献中召回相关片段，并交给大模型生成回答。

- **⌨️ 支持命令行交互配置**  
  支持通过命令行输入 Zotero 路径、API Key、Base URL、模型名称等配置。

- **🛠️ 为后续增量同步预留结构**  
  已加入 `manifest.py` 模块，用于保存文献库扫描快照，为未来的增量索引与同步做准备。

---

## 🛠️ 环境准备与依赖安装

请确保你的电脑已安装 Python，建议版本为 **3.10** 左右。  
在终端中运行以下命令安装核心依赖：

```bash
python -m pip install -U langchain langchain-community langchain-core langchain-text-splitters langchain-openai faiss-cpu pypdf tqdm
```

如果你使用的是 conda，也可以先创建环境：

```bash
conda create -n askmyzotero python=3.10 -y
conda activate askmyzotero
python -m pip install -U langchain langchain-community langchain-core langchain-text-splitters langchain-openai faiss-cpu pypdf tqdm
```

---

## ⚙️ 核心配置方式

当前项目支持 **OpenAI 风格参数** 配置，主要包括：

- `API_KEY`
- `BASE_URL`
- `chat_model`
- `embedding_model`
- `zotero_path`

你可以通过两种方式提供这些配置：

### 1. 命令行交互输入

```bash
python main.py --interactive-config
```

程序会主动提示你输入：

- Zotero 本地路径
- API Key
- Base URL
- 其他必要配置

### 2. 直接命令行传参

```bash
python main.py ^
  --zotero-path "D:\ZoteroData\storage" ^
  --api-key "sk-xxxxxx" ^
  --base-url "https://your-api-url/v1" ^
  --chat-model "gpt-4o-mini" ^
  --embedding-model "text-embedding-3-small"
```

---

## 🚀 运行说明

### 启动交互式问答

```bash
python main.py --interactive-config
```

### 单次提问

```bash
python main.py ^
  --zotero-path "D:\ZoteroData\storage" ^
  --api-key "sk-xxxxxx" ^
  --base-url "https://your-api-url/v1" ^
  --question "哪些文章提到了检索增强生成？"
```

### 强制重建索引

```bash
python main.py --interactive-config --rebuild
```

---

## 📄 项目目录结构 & 各文件功能说明

```text
AskMyZotero/
├─ .askmyzotero/        # 运行时生成的本地缓存、向量库索引、manifest 等文件
├─ dataset/             # 数据集、测试样例或后续实验文件目录
├─ src/
│  ├─ __init__.py       # 将 src 声明为 Python 包，便于模块导入
│  ├─ config.py         # 统一管理命令行参数、环境变量与运行配置
│  ├─ indexer.py        # 负责向量库构建/加载、检索器与 RAG 问答链
│  ├─ manifest.py       # 负责保存与读取文献库扫描快照，为增量同步做准备
│  ├─ parser.py         # 负责 PDF 解析、文本切块与文本块缓存
│  └─ scanner.py        # 负责扫描 Zotero 本地目录中的 PDF 文件
├─ tmp/                 # 临时文件目录，用于调试、中间结果或实验文件
│  ├─ zotero_rag.py        # 早期单文件版本/实验脚本，可作为历史参考
│  └─ AskMyZotero_README_regenerated.md        # 生成的项目说明文档
├─ .gitignore           # Git 忽略配置
├─ main.py              # 程序主入口，串联配置、扫描、索引与问答流程
└─  README.md            # 项目说明文档
```

---

## 🔄 当前程序执行流程

当前版本的整体流程如下：

1. `main.py` 解析命令行参数并读取配置
2. `scanner.py` 扫描 Zotero 本地 PDF 文件
3. `manifest.py` 保存当前文献库扫描快照
4. `parser.py` 读取 PDF 内容并切分为文本块
5. `indexer.py` 构建或加载本地向量库
6. `indexer.py` 构建 RAG 检索与问答链
7. 用户输入问题后，系统检索相关文献片段并生成回答
