# 借鉴 RAG-Assistant-for-Zotero 的低风险改造方案

## 1. 目标

在不推翻当前 AskMyZotero 架构的前提下，吸收 `aahepburn/RAG-Assistant-for-Zotero` 中最值得借鉴的部分，优先提升：

- 检索召回率
- 检索精排准确率
- 多轮问答可用性
- 回答的可解释性与 groundedness

本方案明确遵守当前项目边界：

- 保持主链路仍为 `scanner -> parser -> indexer -> metadata_store -> qa_agent -> aggregator`
- 优先局部改造，不做大重构
- 默认不删除 `.askmyzotero/`
- 尽量把“不需要重建索引”的改造放前面

---

## 2. 当前项目与对方项目的关键差异

### 我们当前已有的优势

- 有较明确的 `intent routing`
- 有 `metadata_store` 级别的硬过滤
- 有 paper-level aggregation
- 有 section-aware chunk 重排
- 有 prompt log，便于调试

### 对方项目更强的部分

- `dense + BM25 + RRF + cross-encoder rerank`
- follow-up 问题改写为 standalone query
- 更结构化的 academic answer prompt
- 更显式的 grounding / limitations 约束
- snippet diversity control
- 自动抽取 metadata filter 失败时的回退逻辑

---

## 3. 总体改造顺序

建议分四期推进，前两期优先，且都不要求改动底层 chunk 数据结构。

### Phase 1：回答层和重排层增强

目标：

- 在不动索引格式的前提下，提升答案稳定性
- 让 `survey / comparison / definition` 的输出更像学术回答

建议改动：

- `src/qa_agent.py`
- `src/aggregator.py`
- 新增 `src/prompt_templates.py` 或 `src/answer_prompts.py`

本期重点：

- 把现有回答 prompt 从内联字符串中抽离
- 按 `intent` 区分回答模板
- 增加显式 grounding 规则
- 增加 limitations / uncertainty 输出约束
- 增加 chunk diversity 控制

是否需要重建索引：

- 不需要

验证方式：

- 直接使用现有索引跑 `/api/ask`
- 对 `fact_qa / comparison / survey / paper_lookup` 至少各测 3 个问题

### Phase 2：混合检索

目标：

- 补足 dense-only 对专有词、方法名、论文名、缩写的不稳定召回

建议改动：

- `src/indexer.py`
- `src/qa_agent.py`
- 可新增 `src/hybrid_retrieval.py`
- 可新增 BM25 缓存文件到 `.askmyzotero/<index_name>/`

本期重点：

- 基于已有 splits 或 metadata 构建 BM25 语料
- 引入 `dense + sparse` 双路召回
- 使用简单 RRF 融合
- 保持当前 metadata hard filter 不变

是否需要重建索引：

- 不需要重建 FAISS
- 需要额外生成 BM25 缓存

验证方式：

- 用方法名、论文名、缩写、年份约束类问题做对比
- 重点比较 top-k 命中文献是否更稳定

### Phase 3：多轮检索改写

目标：

- 支持 follow-up 问题在检索前变成 standalone query

建议改动：

- `server.py`
- `src/api_models.py`
- `src/qa_agent.py`
- 可新增 `src/query_condenser.py`

本期重点：

- 只在“有上下文且 query 明显依赖前文”时触发 condensation
- 不改变当前单轮问答能力
- 先做后端接口支持，再考虑前端会话状态

是否需要重建索引：

- 不需要

验证方式：

- 连续追问测试，例如：
  - “HippoRAG 2 的 workflow 是什么？”
  - “它和 GraphRAG 的区别是什么？”
  - “那它在实验里有什么优势？”

### Phase 4：自然语言 metadata filter 抽取与回退

目标：

- 让用户不必总是手动传 filter
- 避免误过滤导致 0 结果

建议改动：

- `src/qa_agent.py`
- `src/metadata_store.py`
- 可新增 `src/filter_extractor.py`

本期重点：

- LLM 或规则提取作者、年份、tag、collection、title 限制
- 若自动过滤结果为空，则自动回退为无 filter 检索
- 区分“自动 filter”与“用户显式 filter”

是否需要重建索引：

- 不需要

验证方式：

- “2024 年关于 GraphRAG 的论文有哪些”
- “找标签里有 biology 的 RNA 相关论文”
- 验证自动 filter 出错时是否正确回退

---

## 4. 每一期的落地设计

## 4.1 Phase 1：回答层增强

### 4.1.1 抽离 prompt 模板

当前问题：

- 回答 prompt 在 `qa_agent.py` 和 `aggregator.py` 里内联
- 部分字符串存在编码历史问题
- 不便于按 intent 定制

建议：

- 新增 `src/answer_prompts.py`
- 提供以下函数：
  - `build_direct_answer_prompt(intent, query, context)`
  - `build_paper_list_prompt(intent, query, paper_context)`
  - `build_intent_specific_output_rules(intent)`

最低要求：

- `fact_qa`：短答案 + 关键证据 + 明确 citation
- `comparison`：优先表格或并列结构
- `survey`：总览 + 分点 + limitations
- `definition`：定义 + 来源 + 可能的不同表述
- `paper_lookup`：说明“哪些论文相关”及原因，但不在正文重复大段书目信息

### 4.1.2 增加质量约束

建议在 prompt 中明确写出：

- 每个事实尽量附编号引用
- 不足时明确承认证据不足
- 区分“文献明确说了什么”和“系统综合判断了什么”
- 不要编造来源中没有的实验结果

可借鉴但不必照搬的标签：

- `[FINDING]`
- `[SYNTHESIS]`
- `[GAP]`

建议先不直接暴露这些标签给用户，而是作为内部 prompt 约束。

### 4.1.3 增加 diversity 控制

当前问题：

- top evidence 容易被少数论文占满

建议：

- 在 `_rerank_chunks_for_query` 之后增加“每篇论文最多保留 N 条 chunk”的控制
- broad query 与 focused query 分开：
  - broad：每篇最多 2-3 条，总共 6-8 条
  - focused：每篇最多 4-5 条，总共 8-10 条

可新增函数：

- `select_diverse_evidence_chunks(...)`

放置位置建议：

- `src/qa_agent.py`

---

## 4.2 Phase 2：混合检索

## 4.2.1 保持 metadata_store 为第一层

这一点不要向对方项目回退。

推荐顺序：

1. `metadata_store.resolve_candidate_scope` 先缩小候选集
2. dense 检索和 BM25 检索都只在候选集内做
3. 双路结果做 RRF
4. 再做当前的 query-aware rerank

这会比“全库 BM25 + 全库 dense”更适合本项目。

## 4.2.2 新增 BM25 缓存

建议新文件：

- `src/hybrid_retrieval.py`

建议职责：

- 构建 BM25 文本语料
- 保存 `chunk_id -> tokenized_text`
- 提供：
  - `build_bm25_cache(...)`
  - `load_bm25_cache(...)`
  - `search_bm25(query, candidate_ids, k)`
  - `fuse_rrf(dense_hits, sparse_hits, k=60)`

缓存位置建议：

- `.askmyzotero/<index_name>/bm25_cache.pkl`

BM25 文本建议组成：

- `paper_title`
- `section`
- `chunk text`
- 可选拼上 `authors / tags / collections`

### 4.2.3 RRF 融合策略

建议先用简单实现：

- dense rank score: `1 / (rrf_k + rank)`
- sparse rank score: `1 / (rrf_k + rank)`
- 最终分数求和

第一版不需要 cross-encoder。

原因：

- 改造小
- 无需新增模型依赖
- 便于先验证 hybrid retrieval 是否值得继续投入

### 4.2.4 cross-encoder rerank 放在第二步

对方项目用了 cross-encoder rerank，但这里建议晚一点做。

原因：

- 成本更高
- 会引入新依赖或额外模型下载
- 先验证 BM25 + RRF 已经能解决多少问题

建议做成可选增强项：

- `enable_cross_encoder_rerank: bool = False`

---

## 4.3 Phase 3：多轮检索改写

### 4.3.1 新增 Query Condenser

建议新文件：

- `src/query_condenser.py`

建议职责：

- 判断当前 query 是否是 follow-up
- 基于最近若干轮历史，把 query 改写成 standalone question

建议提供：

- `should_condense(query, history) -> bool`
- `condense_query(query, history, llm) -> str`

### 4.3.2 与现有 rewrite 的关系

顺序建议是：

1. 如果是 follow-up，先做 `condense_query`
2. 再把 standalone query 送进现有 `_rewrite_query_with_llm`

这样职责更清楚：

- condenser 负责补全上下文
- rewrite 负责把检索表达改写得更像学术写作

### 4.3.3 接口边界

当前 `handle_query()` 只接收单个 query。

建议扩展但不破坏兼容：

- `handle_query(query, top_k=None, filters=None, history=None)`

如果 `history is None`：

- 维持当前行为不变

---

## 4.4 Phase 4：自然语言 filter 抽取与回退

### 4.4.1 自动 filter 只做“补充”，不抢用户显式 filter

建议规则：

- 如果 API 已经传入 `filters`，优先使用显式 filter
- 自动抽取的 filter 只在 `filters` 为空时启用

### 4.4.2 建议先规则后 LLM

可先覆盖高价值场景：

- 年份或年份区间
- “某作者的论文”
- “某合集/某标签”
- 标题包含某词

原因：

- 成本低
- 可解释
- 错误更容易控制

### 4.4.3 自动 filter 失败回退

建议仅对“自动抽取 filter”启用回退：

- 自动 filter 命中 0 个候选时
- 再做一次无 filter 检索
- 在 debug 中标记 `auto_filter_fallback = true`

用户显式 filter 不应自动去掉。

---

## 5. 文件级改造建议

### `src/qa_agent.py`

建议改动：

- 接入 `answer_prompts`
- 新增 evidence diversity 控制
- 接入 hybrid retrieval 调度
- 接入 query condensation
- 区分显式 filter 与自动 filter

优先级：

- 最高

### `src/aggregator.py`

建议改动：

- 保留 paper 聚合逻辑
- 只重写回答 prompt 构造部分
- 让 `paper_lookup` 与 `survey` 的生成格式更稳定

优先级：

- 高

### `src/metadata_store.py`

建议改动：

- 尽量少动核心数据结构
- 增加对“自动 filter 回退”所需的诊断信息支持即可

优先级：

- 中

### `src/indexer.py`

建议改动：

- 尽量不动 FAISS 主索引逻辑
- 如需 BM25 构建，优先基于 splits 或 metadata 额外生成缓存

优先级：

- 中

### `server.py`

建议改动：

- 如果后续支持多轮 history，需要扩展 `/api/ask` 请求结构

优先级：

- Phase 3 再做

---

## 6. 推荐实施顺序

## 6.1 第一周

- 抽离回答 prompt
- 按 intent 区分 direct answer / paper list 模板
- 增加 evidence diversity 控制
- 保留现有检索逻辑不变

预期收益：

- 回答更稳
- citation 和 limitations 表达更清晰

## 6.2 第二周

- 新增 BM25 缓存
- 接入 dense + BM25 + RRF
- 保留现有 query rewrite 和 hard filter

预期收益：

- 方法名、论文名、缩写类问题命中更稳

## 6.3 第三周

- 引入 follow-up condenser
- 扩展接口支持 history

预期收益：

- 多轮问答可用性显著提升

## 6.4 第四周

- 做自然语言 filter 抽取
- 增加自动 filter 回退

预期收益：

- 更像真正可对话的 Zotero 助手

---

## 7. 验证清单

### 回答质量

- `fact_qa` 是否更少编造
- `comparison` 是否更适合并列比较
- `survey` 是否有清晰总览
- `paper_lookup` 是否能说明“为什么这些论文相关”

### 检索质量

- 专有方法名是否更容易命中
- 论文名缩写是否更稳定
- metadata 限制是否仍然生效
- broad query 是否不再被单篇论文霸占

### 会话能力

- follow-up 是否能正确解析代词
- standalone rewrite 是否没有明显偏题

### 回退机制

- 自动 filter 误判时是否能自动恢复结果
- 用户显式 filter 时是否不会被偷偷放宽

---

## 8. 风险与控制

### 风险 1：Prompt 变重，答案变长

控制：

- 按 intent 区分模板
- 不对所有问题都强制四段式输出

### 风险 2：BM25 引入噪声

控制：

- 先只做 RRF，不立即引入 cross-encoder
- 保留当前 query-aware rerank 作为第二层精排

### 风险 3：会话改写偏题

控制：

- 只在明显 follow-up 场景启用
- 保留原 query 作为 fallback

### 风险 4：自动 filter 误伤召回

控制：

- 自动 filter 与显式 filter 区分
- 自动 filter 结果为空时回退

---

## 9. 建议的第一个可执行任务

如果只做一个最小闭环，建议先做：

1. 抽离回答 prompt
2. 为 `comparison / survey / definition` 增加 intent-specific 输出模板
3. 增加 evidence diversity 控制

原因：

- 风险最小
- 不改索引结构
- 可以立即用现有索引验证
- 能先解决“答案组织方式”问题，再继续处理“检索召回”问题

---

## 10. 结论

对方项目最值得借鉴的不是单个 prompt，而是“多阶段检索 + 多轮改写 + 更严格的回答约束”这三件事。

对 AskMyZotero 来说，最合理的落地方式不是照搬其 Chroma 架构，而是：

- 保留当前 `metadata_store + FAISS + paper aggregation`
- 在此基础上补上 `BM25/RRF`
- 再补上 `follow-up condensation`
- 最后把回答 prompt 做成 intent-aware、grounded、可解释的模板体系

这样改造风险最低，也最符合当前仓库已有设计。
