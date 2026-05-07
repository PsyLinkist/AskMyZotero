# AskMyZotero

AskMyZotero 是一个面向本地 Zotero 文献库的 RAG 问答项目。  
它会扫描本地 Zotero `storage` 下的 PDF，补齐 Zotero sqlite 中的文献元数据，完成切块、向量化、索引构建，并提供带引用的问答与文献检索能力。

当前项目重点不是“通用聊天”，而是围绕个人文献库场景做可控检索，包括：

- 本地 PDF 扫描与元数据绑定
- section-aware chunk 切分
- FAISS 向量检索
- `metadata.db` 硬过滤候选集
- query intent / rewrite / retrieval routing
- paper 级聚合与证据组织
- FastAPI + HTML 前端交互

## 文档入口

- [个人文献库问答系统](https://docs.qq.com/doc/DWXRMZWZzcmN5WEZh)
- [个人文献库问答系统阶段二](https://docs.qq.com/doc/DWW5PQmxXSkpzRHJB?nlc=1)
- [问答测试数据集及分析](https://docs.qq.com/doc/DWWJhdFJPanpjYnBx?aidPos=detail&no_promotion=1&is_blank_or_template=blank&nlc=1)
- [项目贡献](https://docs.qq.com/doc/DWUZJVXdHVGhDb0JK?no_promotion=1&is_blank_or_template=blank)

## 当前能力

- 支持扫描 Zotero `storage` 中的 PDF，并补齐 `title / authors / tags / collections / year / venue`
- 支持基于正文段落与 section 的 chunk 切分，并生成稳定 `chunk_id`
- 支持构建、加载和重建本地 FAISS 向量索引
- 支持基于 SQLite 元数据库进行硬过滤候选集裁剪
- 支持 `fact_qa / comparison / definition / survey / paper_lookup` 五类查询意图
- 支持 query rewrite、检索模式推断和答案样式推断
- 支持将 chunk 级命中聚合为 paper 级结果，并输出证据片段与引用
- 支持结构化学术回答 prompt，输出直接回答、关键证据、综合判断与必要的证据边界说明
- 支持引用编号清洗与重映射，避免无依据地强行补齐所有候选文献引用
- 支持 Web 前端筛选条件：
  - `Collection`
  - `Tag`
  - `Author`
  - `Date range`
  - `Results(top-k)`
- 支持在前端打开本地 PDF 文件
- 支持后台重建索引与进度查询
- 支持显式增量更新索引：
  - 命令行 `python main.py --incremental`
  - 前端主页面右上角“同步”按钮
  - 后端 `/api/sync` 后台任务与 `/api/sync/status` 状态查询
- 增量更新采用 tombstone 策略：新增/修改 PDF 追加新向量，删除/修改旧 chunk 从 active metadata 中移除，检索时过滤 inactive chunk

## 检索链路

当前主链路如下：

1. `scanner.py`
   扫描 Zotero 附件目录，并从 Zotero sqlite 中读取附件对应的父条目元数据。
2. `parser.py`
   清洗 PDF 文本、识别 section、按段落生成 chunk，并补齐 chunk 元数据。
3. `indexer.py`
   构建或加载 FAISS 向量库，并同步重建 `metadata.db`。
4. `metadata_store.py`
   维护 `papers / chunks / paper_authors / paper_tags / paper_collections`，在检索前裁剪候选范围。
5. `qa_agent.py`
   负责 query 解析、意图识别、query rewrite、检索调度、结果整理。
6. `aggregator.py`
   将 chunk 级结果聚合到 paper 级，并按证据质量、section 权重和匹配理由进行重排。
7. `server.py`
   将问答结果、引用、证据片段和调试信息返回给前端。

### 意图识别与回答样式

当前支持五类 intent：

- `fact_qa`：回答具体事实、机制、过程、结果、数字或实现细节。
- `definition`：解释术语、概念或方法定义。
- `comparison`：比较多个概念、方法、论文或结果。
- `survey`：做综述、概览、主题梳理、流程或趋势总结。
- `paper_lookup`：查找、列举、推荐或识别相关论文/文献。

意图识别采用三段式：

1. 先用确定性结构规则处理明显问题。
2. 规则无法判断时调用 LLM 分类，并要求按用户期望的输出形态判断，而不是只看关键词。
3. LLM 失败或返回非法 JSON 时再回退到普通关键词规则。

回答样式由 intent 决定：

- `paper_lookup` 走 paper 级聚合，返回候选论文和对应证据。
- 其他问题默认走 direct answer，直接基于 top chunks 组织答案。

### 引用与 Sources 展示

答案中的 `[n]` 引用编号会映射到前端右侧 `Sources` 面板。当前策略是不再为了展示更多候选而强行追加 `[1][2][3]...`；模型只应引用真正支撑正文判断的证据。

需要注意：检索/聚合出来的候选论文数量可能多于正文实际引用数量。候选论文用于召回和排序，正文引用代表最终回答实际采纳的证据。

## 启动与测试

### 启动 exe

运行：

```bash
dist\AskMyZotero.exe
```

### 本地开发启动

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 在项目根目录启动 Web 服务：

```bash
python server.py
```

3. 浏览器访问：

- 主界面：`http://127.0.0.1:8000/`
- 设置页：`http://127.0.0.1:8000/settings.html`
- 健康检查：`http://127.0.0.1:8000/health`

### 命令行模式

```bash
python main.py
```

常见形式：

```bash
python main.py --question "这篇论文的方法分几个阶段？"
python main.py --rebuild
python main.py --incremental
```

### 一键打包

按 `build_exe.bat` 说明在已激活环境中执行，会自动清理旧包并重新打包，生成：

```text
dist\AskMyZotero.exe
```

## 配置说明

配置来源优先级如下：

1. 命令行参数
2. 环境变量
3. `config.yaml` 或 `ASKMYZOTERO_CONFIG` 指向的 YAML

常用配置项：

- `zotero_path`
- `api_key`
- `base_url`
- `chat_model`
- `embedding_model`
- `work_dir`
- `index_name`
- `chunk_size`
- `chunk_overlap`
- `top_k`

Web 模式下，运行时配置默认会写入：

```text
%APPDATA%\AskMyZotero\config.yaml
```

如果设置了 `ASKMYZOTERO_CONFIG`，则优先使用该路径。

## 首次使用与“模拟新用户”

### 首次保存配置与向量化

首次填写设置并点击保存后：

- 后端会写入配置并初始化 Agent
- 如果本地还没有向量索引，会调用嵌入模型完成建库
- 终端会持续输出扫描、解析、向量化和索引构建进度

请等待首次建库完成后再开始检索。

### 模拟新用户

如果要复现“第一次使用”的状态，建议清理下面两类内容。

1. 运行时配置文件

- 默认位置：`%APPDATA%\AskMyZotero\config.yaml`
- 如果设置了 `ASKMYZOTERO_CONFIG`，则删除该环境变量指向的配置文件
- 也可以直接删除整个 `%APPDATA%\AskMyZotero`

2. 索引与切块工作目录

默认相对 `work_dir + index_name`，常见位置：

- 脚本启动：项目根目录下的 `.askmyzotero\default\`
- exe 启动：`AskMyZotero.exe` 同目录下的 `.askmyzotero\default\`

删除对应的 `default` 子目录或整个 `.askmyzotero`，即可清空向量索引和缓存。

## 索引与缓存产物

工作目录通常位于：

```text
.askmyzotero/<index_name>/
```

常见产物包括：

- `faiss_index/`
  向量索引
- `zotero_splits_cache.pkl`
  切块缓存
- `manifest.json`
  扫描快照
- `metadata.db`
  元数据库

说明：

- 除非明确要全量重建，否则优先复用这些产物
- 如果修改了 `scanner.py`、`parser.py`、`metadata_store.py` 的数据结构，通常需要重新生成相关缓存或索引
- 日常新增、删除或修改 Zotero PDF 后，可以优先使用前端“同步”按钮或 `python main.py --incremental` 执行增量更新
- 当前增量更新不会物理删除 FAISS 中的旧向量，而是通过 `metadata.db` active chunk 集合过滤；长期运行后如果 inactive 向量过多，后续会引入 compact rebuild

## API 概览

当前服务端主要接口包括：

- `GET /`
  主问答页面
- `GET /settings.html`
  设置页
- `GET /health`
  服务健康检查
- `GET /api/config`
  读取当前配置摘要
- `POST /api/config`
  保存配置并尝试初始化后端
- `POST /api/init`
  手动初始化后端
- `POST /api/ask`
  执行问答
- `POST /api/open_local_file`
  打开本地 PDF
- `POST /api/rebuild`
  后台触发重建
- `GET /api/rebuild/status`
  查询重建状态和进度
- `POST /api/sync`
  后台触发增量同步索引
- `GET /api/sync/status`
  查询增量同步状态和进度

## 项目结构

```text
AskMyZotero/
├── README.md
├── AGENTS.md
├── Subagents.md
├── config.yaml
├── config.dist.yaml
├── requirements.txt
├── launcher.py
├── main.py
├── server.py
├── settings.html
├── zotero_rag_ui.html
├── docs/
│   ├── AskMyZotero_需求文档.md
│   ├── TODO-检索方法优化.md
│   ├── 开发TODO-checklist-各种基本功能.md
│   ├── 开发日志.md
│   ├── 问题类型支持现状梳理.md
│   ├── 个人文献库常见问题类型.md
│   ├── 20260423_全项目检索优化Roadmap.md
│   ├── 20260423_paper_lookup问题与改造checklist.md
│   ├── 20260423_chunk限制与关键chunk选择TODO.md
│   ├── 20260503_基线评测与检索瓶颈.md
│   ├── 20260501_后续目标checklist.md
│   ├── 20260505_借鉴RAG-Assistant改造方案.md
│   └── 20260507_增量更新索引方案.md
├── src/
│   ├── aggregator.py
│   ├── api_models.py
│   ├── config.py
│   ├── domain_models.py
│   ├── indexer.py
│   ├── manifest.py
│   ├── metadata_store.py
│   ├── parser.py
│   ├── prompt_logger.py
│   ├── qa_agent.py
│   ├── scanner.py
│   └── __init__.py
├── testpdf/
├── dataset/
├── temp/
└── .askmyzotero/
```

## 核心模块说明

### 根目录入口

- `server.py`
  Web 服务主入口，负责页面、配置、健康检查、问答、重建索引和打开本地文件。
- `main.py`
  命令行入口，负责配置解析、索引准备和交互问答。
- `launcher.py`
  用于启动本地服务并打开浏览器。

### `src/`

- `config.py`
  聚合命令行、环境变量和 YAML 配置，生成统一 `AppConfig`。
- `scanner.py`
  扫描 PDF 并读取 Zotero sqlite 中的附件与父条目元数据。
- `parser.py`
  负责 PDF 清洗、段落切分、section 识别、chunk 构建与缓存。
- `indexer.py`
  负责嵌入模型、聊天模型、FAISS 索引构建与加载。
- `metadata_store.py`
  负责 `metadata.db`，用于检索前的硬筛选候选集收缩。
- `qa_agent.py`
  当前问答主流程核心，负责 query intent、query rewrite 和检索调度。
- `aggregator.py`
  将 chunk 结果聚合到 paper 级，并组织证据与答案输出。
- `domain_models.py`
  定义 `QueryBundle`、`PaperCandidate`、`AnswerPayload` 等内部模型。
- `api_models.py`
  定义 FastAPI 的请求与响应模型。
- `manifest.py`
  管理扫描快照，降低不必要的重复建库。
- `prompt_logger.py`
  保存问答调试日志和提示词信息。

## 调试与排查

服务端当前会输出 `[HardFilter]` 相关日志，用来确认每次检索的实际候选范围。  
常见字段包括：

- `candidate_source`
- `candidate_papers`
- `candidate_chunks`
- `total_index_docs`
- `raw_hits`
- `kept_hits`
- `final_hits`

如果问答效果异常，优先按这个顺序排查：

1. `qa_agent.py`
   看意图识别、rewrite 和 retrieval routing 是否偏题。
2. `metadata_store.py`
   看过滤条件是否过严，候选 paper / chunk 是否被过早裁掉。
3. `parser.py`
   看切块是否过碎、过长或 section 识别失真。
4. `aggregator.py`
   看 paper 聚合排序、证据选择、引用组织是否合理。
5. `server.py`
   看 API 层是否丢字段或映射错误。

## 当前边界

- 当前主要实现的是“显式硬筛选 + 向量检索 + 聚合回答”
- LLM 隐式过滤条件解析和多层过滤合并还不完整
- `Archive` 与 `Item type` 还未真正接入后端数据层
- 前端筛选当前仍以输入式交互为主，自动补全能力尚未完善
- 仓库测试体系还不完整，很多验证仍依赖人工链路检查

# 作业要求文档，警钟长鸣
![](https://raw.githubusercontent.com/PsyLinkist/LearningBlogPics/main/20260416112442354.png)
![](https://raw.githubusercontent.com/PsyLinkist/LearningBlogPics/main/20260416112511464.png)
![](https://raw.githubusercontent.com/PsyLinkist/LearningBlogPics/main/20260416112550479.png)
