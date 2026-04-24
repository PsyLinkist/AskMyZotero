# AGENTS.md

本文件定义 AskMyZotero 仓库内通用 Agent 的工作方式。目标不是重复通用编码常识，而是把这个项目里真正重要的约束、入口、边界和协作偏好写清楚，让后续人或 Agent 能快速接手。

## 1. 项目目标

AskMyZotero 是一个面向本地 Zotero 文献库的 RAG 问答系统：

- 扫描本地 Zotero `storage` 下 PDF。
- 从 Zotero sqlite 中补齐题名、作者、标签、合集、年份等元数据。
- 解析 PDF 并生成 section-aware chunks。
- 建立 FAISS 向量索引与 `metadata.db`。
- 对用户问题做意图识别、查询改写、检索、聚合与带引用回答。
- 提供 `FastAPI + HTML` 前端交互界面。

## 2. 代码地图

### 根目录入口

- `server.py`
  - Web 服务主入口。
  - 提供页面、配置保存、健康检查、问答、重建索引等接口。
  - 当前项目的主运行入口应优先从这里理解。

- `main.py`
  - 命令行模式入口。
  - 串联配置解析、索引准备和交互式问答。

- `config.yaml` / `config.dist.yaml`
  - 配置文件。
  - 实际运行时还可能通过 `ASKMYZOTERO_CONFIG` 指向用户目录配置。

### `src/` 核心模块

- `config.py`
  - 聚合命令行、环境变量、YAML，产出 `AppConfig`。
- `scanner.py`
  - 扫描 PDF，读取 Zotero sqlite 元数据，构造 authors/tags/collections/year 等信息。
- `parser.py`
  - 负责 PDF 清洗、分段、section 识别、chunk 组装与缓存。
- `indexer.py`
  - 负责嵌入模型、聊天模型、FAISS 向量库构建/加载、元数据库重建。
- `metadata_store.py`
  - 负责 `metadata.db`，支持检索前硬过滤候选集。
- `qa_agent.py`
  - 当前问答主流程核心。
  - 包含 query intent、query rewrite、检索调度、答案组织。
- `aggregator.py`
  - 将 chunk 结果聚合到 paper 级别，并生成更适合展示的回答结果。
- `manifest.py`
  - 管理扫描快照，减少不必要重建。
- `api_models.py`
  - FastAPI 请求/响应模型。
- `prompt_logger.py`
  - 记录调试日志与提示词信息。

### 非代码目录

- `docs/`
  - 需求、路线图、问题清单、开发日志等中文设计资料。
- `testpdf/`
  - 当前用于实验/评估的论文 PDF 集。
  - 子代理的数据集生成工作默认只面向这里的样本。
- `.askmyzotero/`
  - 本地运行生成的缓存、索引、manifest、metadata.db 等产物目录。

## 3. 默认工作原则

- 先读现有实现，再决定改动方案，不凭印象重写。
- 优先延续当前架构：`scanner -> parser -> indexer -> metadata_store -> qa_agent -> aggregator`。
- 改动尽量局部、可验证、可回滚，不做与当前任务无关的大重构。
- 用户未明确要求时，不主动重建整库索引，不删除缓存，不改生产配置。
- 若任务涉及问答效果，优先检查：
  - query 意图识别
  - 元数据过滤
  - chunk 质量
  - paper 聚合逻辑
  - 引用编号与证据映射

## 4. 配置与运行约定

### 配置来源优先级

由 `src/config.py` 可知，配置主要来自：

1. 命令行参数
2. 环境变量
3. `config.yaml` 或 `ASKMYZOTERO_CONFIG` 指向的 YAML

关键配置包括：

- `zotero_path`
- `api_key`
- `base_url`
- `chat_model`
- `embedding_model`
- `work_dir`
- `index_name`
- `top_k`

### 常用运行方式

- 启动 Web 服务：
  - `python server.py`
- 启动命令行问答：
  - `python main.py`
- 强制重建索引：
  - `python main.py --rebuild`

如果任务会触发大规模向量化或 API 调用，要先确认是否真的需要重建。

## 5. 索引与数据产物

默认工作目录在 `.askmyzotero/<index_name>/`，常见产物：

- `faiss_index/`
- `zotero_splits_cache.pkl`
- `manifest.json`
- `metadata.db`

处理这类文件时遵循：

- 优先复用已有缓存。
- 除非任务明确要求，不删除用户已有索引。
- 如果修改 `parser.py`、`scanner.py`、`metadata_store.py` 的数据结构，需要说明是否需要重建缓存/索引。

## 6. 对问答链路的修改建议

涉及问答效果时，建议按以下顺序排查：

1. `qa_agent.py`
   - 意图分类是否合理
   - rewrite 是否偏题
   - 检索计划是否匹配问题类型
2. `metadata_store.py`
   - filters 是否真正生效
   - 候选 paper / chunk 是否被过早裁剪
3. `parser.py`
   - chunk 是否过碎、过长、错分 section
4. `aggregator.py`
   - 聚合排序、证据选择、引用编号是否合理
5. `server.py`
   - API 层是否丢字段、错映射、错过滤

不要一上来只调 Prompt；该项目很多问题更可能来自检索链路与数据准备。

## 7. 文档与代码改动边界

- 小改动优先补到对应模块，不额外发散出新层次抽象。
- 新文档优先放根目录或 `docs/`，名称应直观。
- 如果只是整理规范、任务说明或协作方式，优先写 Markdown，不要引入新依赖。
- 如果新增脚本用于一次性实验，默认放在 `temp/` 或明确的实验目录，并写明用途。

## 8. 测试与验证要求

本仓库当前没有成熟的自动化测试结构，因此每次改动至少应做下面之一：

- 运行与任务直接相关的脚本或接口。
- 用最小样本验证核心路径。
- 明确说明“未运行测试”的原因。

优先的人工验证路径：

- `python server.py`
- 打开 `/health`
- 如任务与问答有关，再验证 `/api/ask`

如果改动影响 PDF 解析、索引或元数据，需要说明是否建议用 `testpdf/` 进行一次小规模验证。

## 9. 本仓库推荐的提交风格

- 先完成一个完整的小目标，再提交。
- 提交说明聚焦行为变化，不写空泛标题。
- 说明中优先回答：
  - 改了什么
  - 为什么改
  - 是否需要重建索引/缓存
  - 如何验证

## 10. 不应做的事

- 不要未经确认就清空 `.askmyzotero/`。
- 不要未经确认就改 `config.yaml` 中用户真实凭据或路径。
- 不要把一次性实验逻辑直接混入主链路。
- 不要在没有证据的情况下，把问题全部归因到模型或 Prompt。
- 不要新增与当前项目不一致的复杂框架层。

## 11. 与 Skills / Subagents 的关系

- `AGENTS.md`
  - 规定“在这个仓库里做事”的通用方式。
- `Skills`
  - 规定“某一类任务”的专门做法。
- `Subagents`
  - 规定“需要拆分出去的特定角色/线程”。

当前仓库如需拆分子任务，先看 `Subagents.md`。没有列出的子代理，默认由主 Agent 自己完成。
