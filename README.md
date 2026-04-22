# AskMyZotero

一个将 Zotero 文献库作为个人外挂知识库使用的问答项目。  
它会扫描本地 Zotero PDF、提取正文内容与附件元数据、构建向量索引，并支持知识库问答与文献检索两类能力。

## 文档入口

- [个人文献库问答系统](https://docs.qq.com/doc/DWXRMZWZzcmN5WEZh)
- [个人文献库问答系统阶段二](https://docs.qq.com/doc/DWW5PQmxXSkpzRHJB?nlc=1)
- [问答测试数据集及分析](https://docs.qq.com/doc/DWWJhdFJPanpjYnBx?aidPos=detail&no_promotion=1&is_blank_or_template=blank&nlc=1)
- [项目贡献](https://docs.qq.com/doc/DWUZJVXdHVGhDb0JK?no_promotion=1&is_blank_or_template=blank)

## 启动与测试

### 启动

运行 `dist\AskMyZotero.exe`

### 模拟新用户（需要清空“缓存”）

全新用户本机没有写入过的运行时配置和文献索引工作目录。要复现“第一次使用”，建议删掉下面两类位置：

1. 运行时配置文件  
默认：`%APPDATA%\AskMyZotero\config.yaml`  
若设置了 `ASKMYZOTERO_CONFIG` 环境变量，则删除指向的那份配置文件。  
也可删除整个 `%APPDATA%\AskMyZotero` 文件夹以彻底清空本轮用户配置。

2. 索引与切块工作目录  
默认相对 `work_dir + index_name`，一般位于：

- 脚本启动：项目根目录下的 `.askmyzotero\default\`
- exe 启动：`AskMyZotero.exe` 同目录下的 `.askmyzotero\default\`

删除整个 `default` 子目录或整个 `.askmyzotero` 即可清空向量库与缓存。

### 首次保存配置与向量化

首次填写设置并点击保存后：

- 后端会写入配置并初始化 Agent
- 若本地没有向量索引，会调用嵌入模型建库
- 终端会持续输出解析、向量化和索引构建进度

请等待向量与索引构建完成后再开始检索。

### 本地开发测试

1. 激活 Python 环境后，在项目根目录执行：

```bash
python server.py
```

2. 浏览器打开：

- 主界面：`http://127.0.0.1:8000/`
- 设置页：`http://127.0.0.1:8000/settings.html`

### 一键打包

按 `build_exe.bat` 说明在已激活环境中执行，会自动清理旧包并重新打包，得到 `dist\AskMyZotero.exe`。

## 项目结构

```text
AskMyZotero/
├── README.md                                  # 项目说明文档
├── config.yaml                                # 本地运行配置
├── config.dist.yaml                           # 默认配置模板
├── launcher.py                                # 启动 Web 服务并打开浏览器
├── main.py                                    # 命令行入口
├── server.py                                  # FastAPI 服务入口
├── settings.html                              # 设置页面
├── zotero_rag_ui.html                         # 主问答页面
├── docs/
│   ├── AskMyZotero_需求文档.md                # 需求文档
│   ├── TODO-检索方法优化.md                   # 检索方法优化记录
│   ├── 个人文献库常见问题类型.md              # 常见问题归类文档
│   ├── 开发TODO-checklist-各种基本功能.md      # 开发 checklist
│   └── 开发日志.md                            # 持续更新的开发日志
├── src/
│   ├── aggregator.py                          # chunk 聚合、论文排序与答案整理
│   ├── api_models.py                          # FastAPI 请求/响应模型
│   ├── config.py                              # 配置解析与 AppConfig 定义
│   ├── domain_models.py                       # QueryBundle、PaperCandidate 等内部模型
│   ├── indexer.py                             # FAISS 向量索引构建与加载
│   ├── manifest.py                            # manifest 快照读写
│   ├── metadata_store.py                      # SQLite 元数据库与硬筛选候选集
│   ├── parser.py                              # PDF 解析、切块与 chunk 元数据补齐
│   ├── prompt_logger.py                       # 问答日志与调试信息落盘
│   ├── qa_agent.py                            # 问答 Agent 主流程
│   ├── scanner.py                             # PDF 扫描与 Zotero sqlite 元数据读取
│   └── __init__.py
└── temp/                                      # 临时文件与历史生成物
```

## 各文件作用

### 根目录

- `launcher.py`
  - 启动 Web 服务并打开浏览器

- `main.py`
  - 命令行入口

- `server.py`
  - FastAPI 服务入口
  - 提供配置、健康检查、问答、重建索引、打开本地文件等接口

- `settings.html`
  - 设置页面
  - 用于配置 Zotero 路径、模型、API 等参数

- `zotero_rag_ui.html`
  - 主问答页面
  - 包含问答区、左侧硬筛选栏、右侧引用面板

- `config.yaml`
  - 本地运行配置

- `config.dist.yaml`
  - 默认配置模板

### `src/`

- `aggregator.py`
  - 将 chunk 级命中聚合为 paper 级候选
  - 负责论文排序、证据整理与答案辅助生成

- `api_models.py`
  - FastAPI 请求/响应模型定义

- `config.py`
  - 配置解析与 `AppConfig` 定义
  - 统一管理索引目录、缓存目录、`metadata.db` 路径等运行参数

- `domain_models.py`
  - 内部数据模型
  - 例如 `QueryBundle`、`PaperCandidate`、`AnswerPayload`

- `indexer.py`
  - FAISS 向量索引的构建、加载与基础 RAG 能力封装
  - 同步触发 `metadata.db` 的重建

- `manifest.py`
  - 扫描结果 manifest 快照的读写

- `metadata_store.py`
  - SQLite 元数据库模块
  - 维护 `papers / chunks / paper_authors / paper_tags / paper_collections`
  - 用于在检索前根据硬筛选生成候选文献和候选 chunk 集

- `parser.py`
  - PDF 解析
  - section-aware 切块
  - chunk 元数据补齐
  - 生成稳定的 `chunk_id`

- `prompt_logger.py`
  - 问答日志与调试信息落盘

- `qa_agent.py`
  - 当前问答 Agent 主流程
  - 负责 query 解析、query rewrite、检索调度、回答生成、调试日志输出

- `scanner.py`
  - 扫描 Zotero storage 中的 PDF
  - 读取 Zotero sqlite 中的父条目元数据
  - 提取 `authors / tags / collections / year`
  - 负责 collection 层级展开

### `docs/`

- `AskMyZotero_需求文档.md`
  - 需求文档

- `TODO-检索方法优化.md`
  - 检索方法优化记录

- `个人文献库常见问题类型.md`
  - 常见问题归类与示例

- `开发TODO-checklist-各种基本功能.md`
  - 当前开发清单与完成状态

- `开发日志.md`
  - 持续更新的开发日志

## 当前项目能力

- 支持本地 Zotero PDF 扫描与切块
- 支持向量索引构建与手动全库重建
- 支持 `fact_qa / comparison / definition / survey / paper_lookup` 五类查询意图
- 支持 query rewrite
- 支持右侧引用面板展示与打开 Zotero / 本地 PDF
- 支持前端硬筛选：
  - `Collection`
  - `Tag`
  - `Author`
  - `Date range`
  - `Results(top-k)`
- 支持 collection 层级匹配

## 当前检索结果链路（简要）

1. `scanner.py` 扫描 Zotero storage，并读取附件元数据
2. `parser.py` 解析 PDF，并生成 section-aware chunks
3. `indexer.py` 构建或加载 FAISS 索引
4. `metadata_store.py` 维护元数据库，并在检索前生成候选集
5. `qa_agent.py` 完成意图识别、query rewrite、候选子集检索与回答生成
6. `aggregator.py` 将 chunk 检索结果聚合为论文级结果
7. `server.py` 返回结果给前端展示

## 索引与缓存产物

工作目录下通常会生成：

- `faiss_index/`
  - 向量索引
- `zotero_splits_cache.pkl`
  - chunk 缓存
- `manifest.json`
  - 扫描快照
- `metadata.db`
  - 元数据库

默认位于：

```text
.askmyzotero/<index_name>/
```

## 调试日志

当前服务端会输出 `[HardFilter]` 日志，用于确认每次请求实际使用的候选集范围。

常见字段包括：

- `candidate_source`
- `candidate_papers`
- `candidate_chunks`
- `total_index_docs`
- `raw_hits`
- `kept_hits`
- `final_hits`

## 当前已知边界

- 当前只完成了前端显式硬筛选层
- LLM 隐式过滤条件解析与两层过滤合并尚未接入
- `Archive` 与 `Item type` 尚未接入真实数据层
- 前端筛选目前为输入式交互，尚未加入自动补全

# 作业要求文档，警钟长鸣
![](https://raw.githubusercontent.com/PsyLinkist/LearningBlogPics/main/20260416112442354.png)
![](https://raw.githubusercontent.com/PsyLinkist/LearningBlogPics/main/20260416112511464.png)
![](https://raw.githubusercontent.com/PsyLinkist/LearningBlogPics/main/20260416112550479.png)
