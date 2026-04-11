# AskMyZotero：基于 Zotero 的个人文献库 RAG + Agent 应用需求文档

> 文档目的：将当前项目从“基于 chunk 的简单 RAG 问答”升级为“面向个人学术文献库的 Hybrid RAG / GraphRAG-lite / Agent 协同系统”，并将工作拆分为可并行开发的任务包。  
> 适用对象：项目负责人、RAG 开发成员、Agent 开发成员。  
> 当前代码基础：`src/parser.py`、`src/indexer.py`、`src/scanner.py`、`src/manifest.py`、`src/config.py`、`main.py`

---

## 1. 项目背景与当前问题

当前系统的基本流程为：

**用户问题 → 全库 chunk 检索 → 取 top-k 片段 → 拼接上下文 → LLM 回答**

该流程对“找相似片段”有效，但对 Zotero 文献库中的典型学术问题支持不足，尤其体现在以下几类问题上：

1. **方法属性检索失败**
   - 示例：哪个文献使用 PPR 算法？
   - 问题：术语缩写 / 全称不一致；缺乏文献级归因；只检索片段，不能稳定判断“哪篇论文使用了某方法”。

2. **对比 / 排除类问题失败**
   - 示例：哪些文献没有使用一般三元组知识图谱，而是用了超图之类的方法？
   - 问题：当前系统擅长“找相似文本”，不擅长“不是 A 而是 B”的结构化判断。

3. **主题集合类问题失败**
   - 示例：我看过的关于 multi-hop retrieval 的工作有哪些？
   - 问题：只返回少量 chunk，缺少文献级聚合和 broad-topic 检索能力。

4. **元数据与可追溯性不足**
   - 示例：回答中只有“片段 4 / 片段 5”，没有完整论文标题、年份、作者、出处。
   - 问题：chunk 元数据不足；生成前缺少文献级聚合。

5. **“主体方法”判断不稳定**
   - 示例：哪些论文提到 GraphRAG，但不是泛泛提及，而是真正的方法主体？
   - 问题：缺少标题 / 摘要 / 方法节权重，无法区分“顺带提及”和“方法主体”。

---

## 2. 项目目标

将系统升级为一个面向 **Zotero 本地文献库** 的个人学术助手，支持以下能力：

### 2.1 检索能力
- 支持元数据检索、全文检索、语义检索、混合检索
- 支持文献级 + 段落级两阶段检索
- 支持 query rewrite / acronym expansion / query routing
- 支持 broad-topic / corpus-level 问题
- 支持轻量图增强检索（第二阶段）

### 2.2 回答能力
- 返回**论文级结果**而非片段编号
- 为每篇结果提供：
  - 标题
  - 作者 / 年份 / venue（能获取则返回）
  - 为什么匹配
  - 证据片段
- 支持单篇总结、多篇归纳、论文对比
- 支持证据不足时的保守回答

### 2.3 Agent 能力
- 将 RAG 作为工具层
- 能根据问题类型调用：
  - 搜索论文
  - 总结单篇
  - 对比多篇
  - 定位原文证据
- 支持后续扩展为多轮追问

---

## 3. 总体技术路线

本项目采用 **渐进式三层升级路线**：

### 第一层：Hybrid Academic RAG
目标：解决当前最核心的问题，即“chunk 检索 + 直接生成”过于简单。

引入：
- 稠密检索（保留现有 FAISS）
- 稀疏检索（BM25 / lexical）
- Hybrid fusion（RRF）
- Query rewrite / query routing
- 文献级聚合与证据化回答

### 第二层：RAPTOR-lite
目标：提升 broad-topic / corpus-level 问题效果。

引入：
- paper summary
- section summary
- summary index
- 多粒度检索（chunk / section / paper）

### 第三层：GraphRAG-lite
目标：增强跨论文、多跳、主题关联类问题。

引入：
- paper-topic-method-entity 图
- anchor paper + 1-hop expansion
- 图扩展候选再重排

---

## 4. 建议后的代码结构

```text
AskMyZotero/
├─ main.py
├─ src/
│  ├─ config.py
│  ├─ scanner.py
│  ├─ manifest.py
│  ├─ parser.py
│  ├─ schema.py
│  ├─ metadata_store.py
│  ├─ router.py
│  ├─ query_rewrite.py
│  ├─ aggregator.py
│  ├─ answerer.py
│  ├─ summary_index.py
│  ├─ graph_index.py
│  ├─ indexer.py
│  └─ retrievers/
│     ├─ dense.py
│     ├─ sparse.py
│     ├─ hybrid.py
│     └─ reranker.py
└─ eval/
   ├─ queries.json
   ├─ run_eval.py
   ├─ metrics.py
   └─ report.md
```

---

## 5. 公共数据结构约定

为支持并行开发，先约定统一输入输出协议。

### 5.1 ChunkRecord
```python
{
  "chunk_id": str,
  "paper_id": str,
  "paper_title": str,
  "authors": list[str] | None,
  "year": int | None,
  "venue": str | None,
  "section": str | None,
  "chunk_type": str,   # abstract/introduction/method/experiment/conclusion/body
  "page_start": int | None,
  "page_end": int | None,
  "text": str,
  "source": str,
  "rel_path": str,
}
```

### 5.2 QueryBundle
```python
{
  "raw_query": str,
  "intent": str,  # term_lookup/topic_list/intersection/contrast_filter/primary_method
  "rewritten_queries": list[str],
  "filters": {
      "year_range": None,
      "collection": None,
      "tags": None,
      "read_state": None
  },
  "search_plan": {
      "use_dense": True,
      "use_sparse": True,
      "need_decomposition": False,
      "need_graph_expand": False
  }
}
```

### 5.3 PaperCandidate
```python
{
  "paper_id": str,
  "title": str,
  "authors": list[str] | None,
  "year": int | None,
  "venue": str | None,
  "score": float,
  "match_reason": list[str],
  "evidences": [
      {
        "chunk_id": str,
        "section": str | None,
        "page": int | None,
        "text": str,
        "score": float
      }
  ]
}
```

### 5.4 AnswerPayload
```python
{
  "answer_text": str,
  "papers": list[PaperCandidate],
  "confidence": float,
  "status": "ok" | "insufficient_evidence" | "retrieval_failed",
  "debug": {
      "intent": str,
      "rewritten_queries": list[str],
      "retrieved_chunk_count": int,
      "retrieved_paper_count": int
  }
}
```

---

## 6. 并行任务拆分

---

# 任务 A：论文解析与元数据增强

## 目标
将当前 `PDF -> page docs -> RecursiveCharacterTextSplitter` 升级为**面向论文场景的结构化切块与元数据增强**。

## 涉及文件
- `src/parser.py`
- `src/scanner.py`
- `src/manifest.py`
- 新增 `src/schema.py`

## 输入
- PDF 文件路径列表
- Zotero storage 中的原始 PDF

## 输出
- `ChunkRecord[]`
- 更新后的 `manifest.json`

## 需要实现的效果
1. 每个 chunk 不再只有 `source / page`
2. 每个 chunk 都有：
   - `paper_id`
   - `paper_title`
   - `section`
   - `chunk_type`
   - `page_start/page_end`
3. 支持后续按照 `paper_id` 聚合
4. 支持 manifest 记录增量状态

## 推荐实现方案

### 方案 A（最简可用）
- `paper_id = rel_path`
- `paper_title = 文件名 stem`
- `section` 用正则从页文本中启发式提取
- `chunk_type` 通过 section 名映射：
  - abstract -> abstract
  - introduction -> introduction
  - method / methodology -> method
  - experiment / evaluation -> experiment
  - conclusion -> conclusion
  - 其他 -> body

### 方案 B（增强版）
- 从第一页中抽标题候选
- 如果有 Zotero metadata，则补作者、年份、venue
- 先做 section merge 再切块，避免切散结构

## Manifest 增强要求
当前 `manifest.py` 只做快照记录，需要增强为：

```json
{
  "sha1": "...",
  "size": 123,
  "mtime": 111,
  "status": "new/changed/unchanged/deleted",
  "indexed_at": "...",
  "summary_ready": false,
  "graph_ready": false
}
```

## 验收标准
- 任意 chunk 都可打印出 `paper_id + title + section + page`
- 修改 PDF 后 manifest 能正确识别 `changed`
- 解析结果能支持后续文献级聚合

## 依赖关系
- 无，可立即开始
- 是任务 B / E / F / G 的基础依赖

---

# 任务 B：Hybrid Retrieval 检索核心

## 目标
将当前单一的 FAISS dense 检索升级为：

**dense + sparse + fusion + rerank**

## 涉及文件
- `src/indexer.py`
- 新增 `src/retrievers/dense.py`
- 新增 `src/retrievers/sparse.py`
- 新增 `src/retrievers/hybrid.py`
- 新增 `src/retrievers/reranker.py`

## 输入
- `ChunkRecord[]`
- `QueryBundle`

## 输出
- `top_chunks`
- 检索日志
- 可选的文献预聚合结果

## 需要实现的效果
1. 保留原有 dense 检索能力
2. 增加 lexical / sparse 检索
3. 支持 hybrid fusion
4. 支持 rerank
5. 为后续 paper aggregation 提供更稳的候选 chunks

## 推荐实现方案

### 方案 A（最简可用）
- dense：继续使用 FAISS
- sparse：使用 BM25
- fusion：RRF（Reciprocal Rank Fusion）
- rerank：规则重排，规则包括：
  - query term overlap
  - acronym expansion hit
  - title hit
  - abstract hit
  - method section hit

### 方案 B（增强版）
- dense top 50
- sparse top 50
- RRF 融合
- 用 cross-encoder 或 reranker 模型做二次重排

## 重点支持的问题类型
- 哪个文献使用 PPR 算法？
- 哪些论文提到 GraphRAG？
- 哪些论文和某方法相关？
- broad-topic 初步召回

## 验收标准
- `PPR` 和 `Personalized PageRank` 两种问法结果更接近
- 术语型问题召回比当前更稳定
- `indexer.py` 不再只依赖 `vectorstore.as_retriever(k=top_k)`

## 依赖关系
- 可与任务 A 并行开发
- 最终接入时依赖任务 A 输出的 chunk metadata 更完整

---

# 任务 C：查询路由与 Query Rewrite

## 目标
让系统能够识别问题类型，并据此调整检索策略，而不是所有问题都走同一条语义搜索链路。

## 涉及文件
- 新增 `src/router.py`
- 新增 `src/query_rewrite.py`
- `src/config.py`
- `src/indexer.py`

## 输入
- 用户原始问题字符串

## 输出
- `QueryBundle`

## 需要实现的效果
系统至少能识别以下 5 类问题：

1. `term_lookup`
   - 哪个文献使用 PPR 算法？

2. `topic_list`
   - 我看过的关于 multi-hop retrieval 的工作有哪些？

3. `intersection`
   - 有没有工作同时提到了时间建模和知识图谱？

4. `contrast_filter`
   - 哪些文献不是普通三元组图谱，而是超图之类的方法？

5. `primary_method`
   - 哪些论文提到 GraphRAG，但是真正把它作为方法主体？

## 推荐实现方案

### 方案 A（最简规则版）
通过关键词规则分类：
- “哪个 / 哪篇 / 哪个文献使用了” -> `term_lookup`
- “有哪些 / 哪些工作 / 哪些论文” -> `topic_list`
- “同时 / 既…又…” -> `intersection`
- “不是…而是…” -> `contrast_filter`
- “真正的方法主体 / 不是泛泛提及” -> `primary_method`

Rewrite 先做词表扩展：
- PPR -> Personalized PageRank
- KG -> knowledge graph
- multi-hop retrieval -> multi-hop QA / bridge reasoning / compositional retrieval
- 时间建模 -> temporal reasoning / temporal knowledge graph

### 方案 B（增强版）
- 通过小模型 / LLM 输出：
  - intent
  - rewritten queries
  - decomposition queries

## 验收标准
- 对任意输入 query，系统能打印 intent
- 能输出改写后的 query 列表
- 不同 intent 能配置不同检索参数

## 依赖关系
- 可与任务 A/B 并行
- 最终由任务 B / E 消费其输出

---

# 任务 D：Agent 工具层与编排

## 目标
让 Agent 不再只是“包装聊天”，而是具备工具调用能力，能够按问题类型调用不同功能模块。

## 涉及文件
- Agent 入口模块
- 新增：
  - `tools/search_papers.py`
  - `tools/summarize_paper.py`
  - `tools/compare_papers.py`
  - `tools/locate_evidence.py`

## 输入
- 用户问题
- `AnswerPayload`

## 输出
- 面向前端的最终回答
- 工具调用日志

## 需要实现的效果
Agent 至少能调度以下工具：

### tool 1：search_papers
输入：
- `query: str`

输出：
- `AnswerPayload`

用途：
- 根据问题先检索候选论文

### tool 2：summarize_paper
输入：
- `paper_id`

输出：
- 单篇论文总结

用途：
- 对某篇论文做摘要、创新点、局限性总结

### tool 3：compare_papers
输入：
- `paper_ids: list[str]`
- `aspects: list[str]`

输出：
- 对比结果

用途：
- 比较两篇或多篇论文在方法、任务、数据集等维度上的异同

### tool 4：locate_evidence
输入：
- `paper_id`
- `question`

输出：
- 定位后的证据段落

用途：
- 支持进一步追问和定位原文

## 推荐实现方案

### 方案 A（推荐）
采用函数调用式 Agent，不做复杂 ReAct：
- 输入问题
- 调用 router
- 决定调用哪个工具
- 返回结构化结果

### 方案 B（增强版）
- 支持多步串联：
  - 先列论文
  - 再追问某一篇
  - 再比较其中几篇

## 验收标准
- Agent 至少能调用 `search_papers()`
- 前端看到的是论文级答案，而不是 chunk 拼接原文
- 能支持简单的多轮追问

## 依赖关系
- 可先用 mock 接口独立开发
- 后续接入任务 E 的真实输出

---

# 任务 E：文献级聚合与证据化回答

## 目标
把“chunk 检索结果”转换成“论文级候选结果”，并生成有证据支持的回答。

## 涉及文件
- 新增 `src/aggregator.py`
- 新增 `src/answerer.py`
- `src/indexer.py`

## 输入
- `top_chunks`
- `QueryBundle`

## 输出
- `PaperCandidate[]`
- `AnswerPayload`

## 需要实现的效果
1. 按 `paper_id` 聚合 chunks
2. 每篇论文输出：
   - 标题
   - 作者 / 年份 / venue（如可得）
   - 为什么匹配
   - 证据片段
3. 不再返回“片段 4 / 片段 5”这种答案
4. 证据不足时，不直接瞎答

## 推荐实现方案

### 方案 A（最简可用）
- 同一论文的 chunk 分数求和
- 命中 title / abstract / method 的 chunk 加权
- 每篇论文保留 top 2~3 证据块
- 按论文排序后生成答案

### 方案 B（增强版）
- 加 section 权重：
  - title > abstract > method > experiment > body > related work
- 定义 `primary_method_score`
- 增加 retrieval confidence evaluator：
  - 候选是否集中
  - 是否存在高质量 section hit
  - 是否缺少关键 query term

## 典型回答模板
```text
问题：哪些论文提到 GraphRAG，但是真正把它作为方法主体？

回答：
1. Paper A (2025)
   - 理由：标题与摘要中明确以 GraphRAG 为核心方法
   - 证据：...
2. Paper B (2025)
   - 理由：方法部分将 GraphRAG 作为主体框架展开
   - 证据：...
```

## 验收标准
- 结果按论文输出，而不是按 chunk 输出
- 每篇候选都有匹配理由与证据
- 证据不足时返回 `insufficient_evidence`

## 依赖关系
- 依赖任务 A、B、C 的基础接口
- 完成后可直接供任务 D 接入

---

# 任务 F：层次摘要索引（RAPTOR-lite）

## 目标
增强 broad-topic / long-document / corpus-level 问题的检索能力。

## 涉及文件
- 新增 `src/summary_index.py`
- `src/manifest.py`
- `src/indexer.py`

## 输入
- 每篇论文的 `ChunkRecord[]`

## 输出
- `paper_summary`
- `section_summaries`
- summary index

## 需要实现的效果
1. 对每篇论文离线生成：
   - 一段 paper summary
   - 若干 section summary
2. 建立独立的 summary index
3. 检索时同时检索：
   - raw chunks
   - section summaries
   - paper summaries

## 推荐实现方案

### 方案 A（最简可用）
- 每篇论文生成 1 个 paper summary
- 每个主要 section 生成 1 个 summary
- 单独建 summary FAISS index

### 方案 B（增强版）
- 组织成两层树：
  - leaf：chunks
  - parent：section summaries
  - root：paper summary

## 重点支持的问题
- 我看过哪些关于 multi-hop retrieval 的工作？
- 有没有工作同时提到了时间建模和知识图谱？
- 某方向的研究脉络是什么？

## 验收标准
- broad-topic 问题能召回更多不同论文
- 检索结果不再严重偏向少量相似 chunk
- 增量更新时只重建受影响论文的 summaries

## 依赖关系
- 依赖任务 A
- 建议在任务 E 基本完成后启动

---

# 任务 G：GraphRAG-lite 增强

## 目标
在已有 hybrid retrieval + paper aggregation 的基础上，增加轻量图扩展能力。

## 涉及文件
- 新增 `src/graph_index.py`
- `src/indexer.py`
- `src/manifest.py`

## 输入
- `PaperRecord`
- `ChunkRecord`
- 抽取的 method / topic / entity / dataset 信息

## 输出
- 图索引
- graph-expanded candidates

## 需要实现的效果
1. 建立轻量图：
   - paper
   - topic
   - method
   - entity
   - dataset
2. 支持：
   - anchor papers 检索
   - 图上 1-hop 扩展
   - 图扩展候选重排

## 推荐实现方案

### 方案 A（推荐）
构建 `paper-topic-method-entity` 图：

节点：
- paper
- topic
- method
- entity
- dataset

边：
- paper -> method
- paper -> topic
- paper -> dataset
- paper <-> entity

检索流程：
1. Hybrid retrieval 找 anchor papers
2. 沿图扩展 1-hop 候选 papers
3. 对扩展结果 rerank

### 方案 B（增强版）
- 增加共现权重
- 增加 query-aware expansion
- 增加方法级别边类型

## 重点支持的问题
- 哪些论文和 HippoRAG 思路接近？
- 哪些工作使用了特殊图结构？
- 哪些工作涉及时间建模 + 知识图谱？

## 验收标准
- 图扩展可开关
- 打开图扩展后，跨论文关联问题覆盖更好
- 不要求首版超过纯 hybrid，但需要带来额外候选和解释路径

## 依赖关系
- 依赖任务 A、B
- 最好在任务 E / F 稳定后开发

---

# 任务 H：评测与案例集

## 目标
为系统建立一个小型但有效的项目内评测集，避免“改了很多，但不知道哪里变好了”。

## 涉及文件
- `eval/queries.json`
- `eval/run_eval.py`
- `eval/metrics.py`
- `eval/report.md`

## 输入
- 真实问题样本
- 标注后的正确论文答案 / 支持证据

## 输出
- 检索指标
- 回答指标
- 错误分析报告

## 题目类型建议
至少覆盖以下 5 类：
1. 方法属性查询
2. 主题集合查询
3. 交集查询
4. 对比 / 排除查询
5. 主体方法判断

## 推荐指标

### 检索层
- paper Recall@k
- chunk Recall@k
- MRR

### 聚合层
- top-k 覆盖论文数
- chunk 是否成功合并到正确 paper

### 生成层
- metadata completeness
- citation support rate
- answer completeness

## 验收标准
- 每次大改后都可自动跑一遍评测
- 能分清错误来自：
  - 检索失败
  - 聚合失败
  - 生成失败

## 依赖关系
- 可从第一天就开始收集题目
- 最终与 A~G 所有任务协同

---

## 7. 任务并行与依赖关系建议

### 第一轮可并行启动
- 任务 A：论文解析与元数据增强
- 任务 B：Hybrid Retrieval
- 任务 C：Router + Rewrite
- 任务 D：Agent 工具层
- 任务 H：评测题集初版

### 第二轮接力
- 任务 E：文献级聚合与证据化回答

### 第三轮增强
- 任务 F：RAPTOR-lite
- 任务 G：GraphRAG-lite

---

## 8. 建议开发顺序

### 最小可上线版本（优先实现）
1. 任务 A
2. 任务 B
3. 任务 C
4. 任务 E
5. 任务 D
6. 任务 H

完成以上后，系统即可从“chunk 问答器”升级为“论文级学术助手”。

### 第二阶段增强
7. 任务 F
8. 任务 G

---

## 9. 第一版统一接口建议

建议在项目中尽快确定以下统一函数接口：

```python
# parser / metadata
load_or_create_corpus(config) -> list[ChunkRecord]

# router
build_query_bundle(raw_query: str) -> QueryBundle

# retrieval
search_chunks(query_bundle: QueryBundle, top_k: int = 50) -> list[dict]

# aggregation
aggregate_to_papers(query_bundle: QueryBundle, chunks: list[dict]) -> list[PaperCandidate]

# answer
generate_answer(query_bundle: QueryBundle, papers: list[PaperCandidate]) -> AnswerPayload

# agent
agent_handle(user_query: str) -> AnswerPayload
```

---

## 10. 每个阶段的预期效果

### 阶段 1 完成后
系统应能够明显改善以下问题：
- “哪个文献使用 PPR 算法？”
- “我看过的 multi-hop retrieval 工作有哪些？”
- “哪些论文提到 GraphRAG，且是真正的方法主体？”
- “有没有工作同时提到了时间建模和知识图谱？”

### 阶段 2 完成后
系统应进一步增强：
- broad-topic / 综述类问题
- 跨论文主题归纳
- 模糊回忆式检索
- 方法 / topic / entity 关联扩展

---

## 11. 风险与注意事项

1. **不要一开始就做重型知识图谱**
   - 当前最缺的是文献级表示、hybrid retrieval、aggregation，而不是 full KG。

2. **GraphRAG-lite 应放在第二阶段**
   - 否则容易投入很大，但首版效果不稳定。

3. **评测要尽早建**
   - 否则无法验证修改是否真的解决了当前展示出来的问题。

4. **接口先定，再并行开发**
   - 推荐优先定下：
     - `ChunkRecord`
     - `QueryBundle`
     - `PaperCandidate`
     - `AnswerPayload`

---

## 12. 当前结论

本项目的最优路线不是“直接做一个大而全的 GraphRAG”，而是：

**先做 Hybrid Academic RAG，再做 RAPTOR-lite，最后补 GraphRAG-lite。**

也就是说：

- 第一阶段重点：  
  **metadata + hybrid retrieval + router + aggregation + answering + agent tools**

- 第二阶段重点：  
  **summary index + light graph expansion**

这样既符合当前代码现状，也最符合你们项目周期与并行开发方式。

---

## 13. 附：推荐参考实现思想（只作方法方向参考）

### A. 第一阶段优先借鉴
- Hybrid Retrieval（dense + sparse + rerank）
- Query Rewrite / Acronym Expansion
- Query Routing
- Retrieval Confidence Gating
- Paper-level Aggregation

### B. 第二阶段可借鉴
- RAPTOR-lite（层次摘要检索）
- GraphRAG-lite（paper-topic/entity 图）
- Light graph expansion
- PPR-style paper relation propagation（可选）

---

**文档结束**
