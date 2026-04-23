# paper_lookup 问题梳理与改造 Checklist

日期：2026-04-23

相关日志：
- `.askmyzotero/logs/prompt_logs/20260423_091825_317182.json`

相关代码：
- `src/qa_agent.py`
- `src/aggregator.py`
- `src/parser.py`

## 1. 背景

当前 `paper_lookup` 链路用于回答这类问题：

- 哪些文献使用了某种方法？
- 哪些论文提到了某个主题？
- 哪些工作在方法或实验中采用了某种技术？

这类问题的核心目标不是回答单个事实，而是：

- 尽可能扩大相关论文候选集
- 同时让最终用于排序和回答的证据 chunk 更贴近 `abstract / method / experiment` 等真正描述方法的小节

结合本次日志分析，当前实现没有很好地兼顾这两个目标。

## 2. 本次暴露的问题

查询样例：

- `哪些文献使用了 RAG 方法？`

日志中的关键现象：

- `retrieved_chunk_count = 141`
- `unique_papers = 24`
- 前 `5` 篇论文占全部 chunk 的 `40.43%`
- 前 `10` 篇论文占全部 chunk 的 `66.67%`

说明：

- 检索结果在 chunk 层面高度集中于少数几篇 “RAG 主题论文”
- 后续聚合又继续放大了这种集中趋势
- 最终答案只列出少量头部论文，导致论文候选覆盖面进一步缩小

### 2.1 现有链路中的主要问题

1. `paper_lookup` 仍然是以 chunk 检索为主，而不是以 paper 候选集扩展为主。
2. 多轮 query rewrite 带来了更多相似 chunk，而不是更多不同论文。
3. chunk 去重只发生在 `chunk_id` 级别，没有在 paper 级别做多样性控制。
4. 聚合阶段采用“同一篇论文命中的 chunk 分数直接累加”，导致长文、综述、benchmark、主题高度相关论文持续占优。
5. section-aware 主要体现在聚合加权，没有前移到“召回更像 method 证据的 chunk”这一步。
6. 标题包含查询词会得到额外加分，这会放大“论文主题就是 RAG”而不是“论文使用了 RAG”的偏差。
7. 相同工作的不同版本可能分别参与排序，例如 arXiv 版和会议版，进一步污染候选集。

## 3. 根因分析

### 3.1 检索目标定义偏了

当前实现更接近：

- 找最像 query 的 chunk
- 再把 chunk 聚合成论文

但 `paper_lookup` 真正需要的是：

- 尽可能扩大相关论文候选集
- 再为每篇论文寻找更像“方法使用证据”的 chunk

也就是：

- 当前是 `chunk-first`
- 更合适的是 `paper-broad-recall + targeted-chunk-evidence`

### 3.2 多轮 rewrite 的收益方向不对

当前三轮实际参与检索的 query：

1. 原始 query
2. 方法部分采用
3. 实验中应用

日志显示：

- 第 1 轮保留 120 个 hits
- 第 2 轮仅新增 10 个 hits
- 第 3 轮仅新增 11 个 hits
- 总计 219 个 hits 只是被去重掉的重复结果

说明：

- rewrite 主要是在头部结果附近反复命中
- 没有显著扩展论文候选集
- 但却引入了更强的小节限定，反而把搜索空间往 method/experiment 的表述方式上收窄了

### 3.3 paper 聚合逻辑会放大重复命中

当前聚合逻辑中，同一论文每多命中一个 chunk，就继续累加分数。

结果是：

- 头部论文由于全文反复出现目标术语
- 能拿到更多 chunk
- 又因为 chunk 更多而继续涨分
- 最终造成“同主题论文刷屏”

这种机制不利于：

- 扩大论文候选覆盖面
- 区分“真正使用该方法”和“只是讨论该方法”

## 4. 改造目标

`paper_lookup` 的改造目标应明确为：

1. 尽可能扩大相关论文候选集，而不是尽快缩小到少数头部论文。
2. 最终参与排序和回答的 chunk，尽量贴近 `abstract / method / experiment` 等方法描述性小节。
3. 让多路检索主要贡献“论文覆盖面”，而不是“重复 chunk”。
4. 让 paper 排名更多由“高质量方法证据”驱动，而不是由“命中 chunk 数量”驱动。
5. 为后续支持“某方法是否被采用 / 是否只是提到 / 是否是 benchmark / 是否是 survey”留下扩展空间。

## 5. 建议的新链路

建议将 `paper_lookup` 重构为“两层异构召回 + paper 级融合”：

### 第一层：Broad Paper Recall

目标：

- 尽可能扩大相关论文候选集

特点：

- 不要求召回的 chunk 就已经是最佳证据
- 重点是让更多相关论文进入候选

建议信号：

- 原始 query 的 dense 检索
- target method 的 lexical 检索
- 标题 / 摘要 / 关键词的匹配
- 同义表达扩展，例如：
  - `RAG`
  - `retrieval-augmented generation`
  - 目标方法的常见变体名

这一层的输出应是：

- 较大的 paper candidate set
- 每篇论文一个粗粒度 paper recall score

### 第二层：Targeted Chunk Evidence Recall

目标：

- 在候选论文内部，找到更贴近方法使用证据的 chunk

特点：

- section-aware 前移
- 检索关注的是“method usage evidence”，而不是泛主题相关性

建议检索桶：

- `broad bucket`
- `abstract bucket`
- `method bucket`
- `experiment bucket`

建议内部 evidence queries：

- `uses <method>`
- `adopts <method>`
- `based on <method>`
- `our framework uses <method>`
- `we employ <method>`
- `retrieval-augmented generation`

这一层的输出应是：

- 每篇论文若干代表性 evidence chunks
- 每个 chunk 记录所属 section、得分、用途

### 第三层：Paper-Level Fusion

目标：

- 用 paper 级别的方式融合多路召回结果

要求：

- 多路检索应扩大 paper 集合
- 不能再通过重复 chunk 持续推高同一篇论文

建议方式：

- 每条 query / 每个 bucket 只允许同一论文贡献有限数量的 chunk
- 每篇论文每个 bucket 只保留 top 1 到 top 2 chunk
- 最终 paper 分数由“广覆盖分 + 方法证据分”组成

建议公式方向：

- `final_score = 0.4 * broad_recall_score + 0.6 * method_evidence_score`

其中：

- `broad_recall_score` 负责扩大候选覆盖
- `method_evidence_score` 负责把真正有方法证据的论文排到前面

## 6. 关键设计原则

### 6.1 宽论文，窄证据

对于 `paper_lookup`：

- 论文候选集尽量宽
- 排序证据尽量窄

不能反过来：

- 候选论文过早缩小
- 证据又很泛

### 6.2 限制单篇论文的 chunk 贡献，不限制整体论文覆盖

需要限制的是：

- 单篇论文参与排序的 chunk 数量

不应该过早限制的是：

- 进入候选集的论文数量

### 6.3 多路检索要贡献“覆盖面”，不是“重复度”

如果第二轮、第三轮只是反复命中第一轮已经有的头部论文，则说明融合方式有问题。

### 6.4 section-aware 应该前移到证据召回阶段

section 权重不应只体现在后处理排序上，也应体现在“召回更像方法描述的 chunk”这一阶段。

## 7. 工程改造 Checklist

以下 checklist 按“先止血，再重构”的顺序整理。

### Phase 0：问题复现与基线固化

- [ ] 固化 `paper_lookup` 的典型测试问题集
- [ ] 将 `.askmyzotero/logs/prompt_logs/20260423_091825_317182.json` 纳入分析样例
- [ ] 增加评估指标：
  - `retrieved_chunk_count`
  - `unique_paper_count`
  - `top_5_paper_chunk_share`
  - `top_10_paper_chunk_share`
  - `final_answer_paper_count`
- [ ] 给 `paper_lookup` 单独输出 debug 摘要，便于观察 paper 覆盖面

### Phase 1：先修正现有链路中最明显的偏差

- [ ] 在 `paper_lookup` 的多轮检索中，新增 paper 级统计：
  - 每轮新增了多少篇新论文
  - 每轮新增的 chunk 中有多少来自已出现论文
- [ ] 在 merge 阶段引入 `paper-level quota`
  - 每轮每篇论文最多保留 `N` 个 chunk
  - 默认建议先从 `N = 2` 开始试验
- [ ] 在最终 merged docs 中增加 `paper diversity rerank`
  - 对已经贡献过较多 chunk 的论文施加轻微衰减
  - 保证更多论文能进入后续聚合
- [ ] 降低标题命中奖励在 `paper_lookup` 中的权重
- [ ] 对明显的 `survey / benchmark / review` 类标题增加轻微惩罚或单独标记

### Phase 2：将当前 rewrite 机制改为“多视角检索”

- [ ] 保留原始 query
- [ ] 不再主要依赖自然语言句式 rewrite
- [ ] 引入内部检索视角结构，例如：
  - `broad_topic`
  - `method_usage`
  - `abstract_claim`
  - `experiment_usage`
- [ ] 为每个视角定义独立 query 构造方式
- [ ] 为每个视角定义独立 section 偏好
- [ ] 为每个视角单独记录召回结果和 paper 覆盖贡献

### Phase 3：重构为“先 broad paper recall，再 targeted chunk evidence”

- [ ] 新增 `paper candidate recall` 层
- [ ] 让候选集构建先在 paper 级完成去重与合并
- [ ] broad recall 阶段优先扩大 paper 候选数，而不是追求 chunk 精度
- [ ] 在候选论文内执行 second-pass chunk retrieval
- [ ] second-pass 只针对高价值 section：
  - `abstract`
  - `method`
  - `experiment`
  - 必要时 `introduction`
- [ ] second-pass 输出每篇论文的代表性 evidence chunks

### Phase 4：重构 paper 聚合评分

- [ ] 将“同一论文所有 chunk 直接累加”改为“分桶聚合”
- [ ] 每篇论文每个 bucket 只取 top 1 到 top 2 chunk
- [ ] 设计新分数组成：
  - `broad_recall_score`
  - `abstract_evidence_score`
  - `method_evidence_score`
  - `experiment_evidence_score`
- [ ] 对后续 chunk 采用衰减加分，而不是线性累加
- [ ] 为相同工作的不同版本增加去重或合并策略：
  - DOI 优先
  - arXiv / conference version 合并
  - title normalization 合并

### Phase 5：增强 evidence 质量

- [ ] 为 evidence chunk 增加更强的 snippet 生成逻辑
- [ ] 优先抽取包含“use / adopt / based on / employ / framework / propose”类模式的句子
- [ ] 降低只包含方法名但没有关系表达的句子权重
- [ ] 区分：
  - 真正使用该方法
  - 只是提到该方法
  - 将该方法作为 baseline
  - 综述或 benchmark 讨论该方法

### Phase 6：答案生成层配套改造

- [ ] LLM 输入的 paper context 由“高质量代表证据”组成，而不是靠头部 chunk 堆积
- [ ] 控制每篇论文进入答案生成的证据数
- [ ] 输出时优先保证论文覆盖面
- [ ] 避免回答阶段只重复列出前几篇头部论文
- [ ] 若合适，支持将结果按类型分组展示：
  - 直接使用该方法的论文
  - 提出该方法变体的论文
  - 主要讨论该方法的论文

## 8. 建议的实现切入点

优先建议从以下位置入手：

### `src/qa_agent.py`

重点改造：

- `_retrieve_docs()`
- query rewrite / query planning 逻辑
- 日志诊断输出

目标：

- 从“多轮 chunk 检索 + chunk 去重”
- 过渡到“多视角召回 + paper 级融合”

### `src/aggregator.py`

重点改造：

- `aggregate_to_papers()`

目标：

- 从“按 chunk 线性累加”
- 过渡到“按 paper / bucket / evidence 受控融合”

### `src/parser.py`

当前基础可复用：

- 已经具备 `section`
- 已经具备 `chunk_type`
- 已经具备 `context_window`

后续关注点：

- 是否需要更强的 method-like 段落标记
- 是否需要在 metadata 中显式标注 `section_priority`

## 9. 第一阶段验收标准

第一阶段不要求彻底重构，但至少应达到：

- [ ] 同类 query 下，`unique_paper_count` 明显上升
- [ ] `top_5_paper_chunk_share` 明显下降
- [ ] 最终答案中的论文覆盖数增加
- [ ] 方法相关 query 的证据 chunk 更集中在 `abstract / method / experiment`
- [ ] 综述/benchmark/纯主题论文不再稳定霸榜

## 10. 暂不建议的做法

以下做法当前不建议直接作为主方案：

- [ ] 单纯继续增加自然语言 rewrite 轮数
- [ ] 只靠增大 `retrieve_k` 解决覆盖问题
- [ ] 只在聚合阶段继续调 section 权重
- [ ] 用更严格的过滤规则过早缩小论文候选集

这些做法更可能带来：

- 更多重复 chunk
- 更强的头部论文集中
- 更小的最终论文覆盖面

## 11. 一句话总结

`paper_lookup` 的正确方向不是“更快缩小论文范围”，而是：

- 先尽可能扩大相关 paper candidate set
- 再在这些候选论文内部，优先抽取更贴近方法描述小节的 evidence chunks
- 最后以 paper 级融合而不是 chunk 数量累加完成排序与回答
