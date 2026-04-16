# AskMyZotero

一个将 Zotero 文献库作为个人外挂知识库使用的问答项目。
它会扫描本地 Zotero PDF、提取正文内容与附件元数据、构建向量索引，并支持知识库问答与文献检索两类能力。

## 文档入口

- [个人文献库问答系统](https://docs.qq.com/doc/DWXRMZWZzcmN5WEZh)
- [个人文献库问答系统阶段二](https://docs.qq.com/doc/DWW5PQmxXSkpzRHJB?nlc=1)
- [问答测试数据集及分析](https://docs.qq.com/doc/DWWJhdFJPanpjYnBx?aidPos=detail&no_promotion=1&is_blank_or_template=blank&nlc=1)
- [项目贡献](https://docs.qq.com/doc/DWUZJVXdHVGhDb0JK?no_promotion=1&is_blank_or_template=blank)

## 项目结构

```text
AskMyZotero/
├── README.md                  # 项目说明文档
├── config.yaml                # 默认配置模板
├── launcher.py                # 启动 Web 服务并打开浏览器
├── main.py                    # 命令行入口
├── server.py                  # FastAPI 服务入口
├── settings.html              # 配置页面
├── zotero_rag_ui.html         # 主问答页面
├── docs/
│   ├── AskMyZotero_需求文档.md  # 历史需求文档
│   └── TODO.md                # 后续开发任务
├── src/
│   ├── aggregator.py          # chunk 聚合、论文排序与回答结果整理
│   ├── api_models.py          # FastAPI 请求/响应 schema
│   ├── config.py              # 配置解析与 AppConfig 定义
│   ├── domain_models.py       # QueryBundle、PaperCandidate 等内部模型
│   ├── indexer.py             # 向量库构建、加载与基础 RAG 链
│   ├── manifest.py            # manifest 快照读写与启动扫描
│   ├── parser.py              # PDF 解析、section 感知切块、上下文窗口构建
│   ├── prompt_logger.py       # 问答日志与调试信息落盘
│   ├── qa_agent.py            # 当前知识库问答 Agent 主流程
│   ├── scanner.py             # PDF 扫描与 Zotero SQLite 元数据读取
│   └── __init__.py
└── temp/                      # 临时文件与历史生成物
```

## 项目功能

- 将整个 Zotero 文献库作为个人知识库进行问答
- 支持 `fact_qa / comparison / definition / survey / paper_lookup` 五类查询意图
- 使用 LLM 做 query rewrite，失败时使用规则兜底
- 检索结果会区分“直接回答”与“相关论文列表”两种输出方式
- 引用展示优先使用命中块 `raw_text`，回答生成使用更宽的 `context_window`

## 当前主流程

1. `scanner.py` 扫描 Zotero storage 并读取附件元数据
2. `parser.py` 解析 PDF，并按 `section 约束 + 段落切块 + 相邻块补上下文` 生成文本块
3. `indexer.py` 构建或加载 FAISS 向量索引
4. `qa_agent.py` 进行意图识别、query rewrite、检索与回答生成
5. `aggregator.py` 对论文检索结果做聚合排序，并按问题类型应用 section 权重
6. `server.py` 将结果返回给前端页面展示

## 当前状态

- section 权重采用“基础权重 + 按问题类型覆盖”的两层策略
- 双栏 PDF 提取后的常见 IEEE 页眉页脚噪声已做一轮清洗
- 文本块缓存会根据 `chunk_strategy` 自动判断是否需要重建
