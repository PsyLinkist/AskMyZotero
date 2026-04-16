# AskMyZotero TODO

本文档用于记录当前项目的开发状态、已完成事项与后续待办。

## 当前目标

将 AskMyZotero 从“简单文献检索器”升级为“以 Zotero 为个人外挂知识库的问答系统”。

目标形态：

- 先回答问题，再展示证据与引用
- 支持不同问题类型的意图识别与不同处理策略
- 检索结果优先围绕“证据”组织，而不只是论文列表
- 逐步提升召回质量、文本质量与前端使用体验

## 已完成

### 1. 主流程重构

- 已新增 `src/qa_agent.py` 作为当前问答主入口
- 已将后端从固定 `paper_search` 主链路切换为“意图识别 + 策略分发”
- 已支持以下查询意图：
  - `fact_qa`
  - `comparison`
  - `definition`
  - `survey`
  - `paper_lookup`

### 2. API 与前端返回结构调整

- 已新增 `src/api_models.py`
- `/api/ask` 已支持返回：
  - `intent`
  - `answer`
  - `answer_type`
  - `evidence_summary`
  - `references`
  - `meta_data`
- 前端主区域已优先展示回答内容，而不是直接展示论文列表

### 3. Query Rewrite

- 已接入 LLM 驱动的 query rewrite
- 当 LLM rewrite 失败时，仍保留规则兜底
- query rewrite 已用于多查询召回

### 4. Section 权重体系

- 已实现“两层 section 权重”机制：
  - 基础权重
  - 按问题类型覆盖权重
- 已在以下路径中使用该权重：
  - 论文聚合排序
  - 直接回答证据筛选
- 已将当前实际生效的 `section_weights` 记录进 debug 日志

### 5. 文本切块与上下文策略

- 已将切块策略升级为：
  - `section 约束`
  - `段落切块`
  - `相邻块补上下文`
- 已取消旧的 chunk overlap 依赖
- 已加入 `context_before / context_after / context_window`
- 已对 `context_window` 做段落级去重拼接

### 6. 展示层与回答层分离

- 引用展示优先使用命中块 `raw_text`
- 回答生成使用更宽的 `context_window`
- 已避免“引用卡片直接展示邻居块拼接结果”导致的大量重复自然段

### 7. PDF 噪声清洗

- 已加入 IEEE 常见页眉页脚与版权信息清洗
- 已对相邻重复段落做基础去重

### 8. 结构收尾

- 已删除旧文件：
  - `src/answerer.py`
  - `src/bootstrap.py`
  - `src/metadata_store.py`
  - `src/paper_agent_v2.py`
  - `src/api_schemas.py`
- 已将：
  - `prepare_manifest_snapshot()` 合并进 `src/manifest.py`
  - `load_attachment_metadata()` 合并进 `src/scanner.py`
- 已将 parser 实现并回 `src/parser.py`

## 正在使用的当前结构

```text
src/
├── aggregator.py      # chunk 聚合、论文排序、回答结果整理
├── api_models.py      # FastAPI schema
├── config.py          # 配置解析
├── domain_models.py   # 内部数据模型
├── indexer.py         # 向量索引构建与加载
├── manifest.py        # manifest 快照
├── parser.py          # PDF 解析、切块、上下文窗口
├── prompt_logger.py   # 调试日志
├── qa_agent.py        # 当前问答 Agent 主流程
├── scanner.py         # PDF 扫描与 Zotero 元数据读取
└── __init__.py
```

## 下一阶段待办

### P0

1. 提升召回质量
- 重点解决：
  - `PPR` 相关问题无法稳定召回 `HippoRAG`
  - 方法名、缩写、别名召回不稳定
- 建议方向：
  - 引入 hybrid retrieval
  - 增加标题 / 元数据 boosting
  - 扩展方法名别名词典

2. 强化“证据驱动回答”
- 目前已经不是单纯论文列表，但回答阶段仍可继续增强
- 建议增加：
  - 更明确的“支持 / 不支持 / 不确定”判断
  - 证据冲突检测
  - 更稳定的 citation 组织方式

3. 过滤或降权 references 区块
- 目前 references 已纳入 section 体系并降权
- 后续建议进一步验证：
  - 是否完全不参与直接回答主证据
  - 是否只作为辅助召回信号保留

### P1

4. 动态推荐词
- 当前推荐词仍需改造成动态生成
- 可基于：
  - 最近搜索
  - 当前库主题
  - 高频术语

5. 自适应检索数量
- 目前仍主要依赖固定 `top_k`
- 后续建议改为：
  - 初召回数量
  - 重排数量
  - 最终证据数量
  三层控制

6. 更强的 PDF 文本清洗
- 当前已处理一部分 IEEE 噪声
- 后续仍需继续优化：
  - 双栏串列
  - 标题断裂
  - 异常空格
  - 页码 / 表格残留

### P2

7. 前端体验优化
- 区分：
  - 答案
  - 证据摘要
  - 引用与相关文献
- 增加更清晰的调试展示与展开上下文能力

8. 日志可读性提升
- 将以下内容组织成更清晰的调试摘要：
  - `intent`
  - `rewritten_queries`
  - `section_weights`
  - 最终命中的 section 分布
  - 最终采用的证据块

## 建议实施顺序

1. 优先做召回优化，先把 HippoRAG / PPR 这类问题拉稳
2. 再增强证据驱动回答质量
3. 然后处理 references 策略与自适应检索数量
4. 最后再补动态推荐词和前端体验优化
