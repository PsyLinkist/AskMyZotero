# chunk 限制与关键 chunk 选择 TODO

日期：2026-04-23

## 目标

- 避免“找论文集合”类 query 被少数论文刷屏
- 避免“单篇论文理解”类 query 因限制 chunk 而信息不全
- 在必须限制 chunk 时，尽量保住关键证据 chunk

## TODO

### 1. 按 query 类型切换 chunk 保留策略

- [ ] 区分两类检索模式：
  - `paper_set_retrieval`
  - `single_paper_understanding`
- [ ] 对 `paper_lookup / survey / 部分 comparison` 启用严格的 per-paper chunk quota
- [ ] 对“这篇论文讲了什么 / 总结这篇论文 / 这篇论文的方法和实验是什么”这类 query 放宽或关闭 per-paper chunk quota

### 2. 单论文理解类 query 改为 section coverage

- [ ] 不再只按 top-k chunk 截断
- [ ] 改为按 section 补齐：
  - `abstract`
  - `method`
  - `experiment`
  - `conclusion`
- [ ] 确保单篇论文理解类 query 能覆盖论文主结构

### 3. 给 chunk 增加“关键性评分”

- [ ] 在 retrieval score 之外加入：
  - `section_bonus`
  - `query_relation_bonus`
  - `keyword_density_bonus`
  - `noise_penalty`
- [ ] 对不同 query 类型配置不同的高价值 section

### 4. 增加关系表达识别

- [ ] 对“使用某方法”类 query，优先保留包含以下模式的 chunk：
  - `we use`
  - `we employ`
  - `we adopt`
  - `based on`
  - `our method`
  - `our framework`
- [ ] 降低以下 chunk 的优先级：
  - 只提到术语但没有关系表达
  - related work
  - baseline 列表
  - 表格噪声
  - references

### 5. 限制 chunk 时按 section 分层保留

- [ ] 不直接取单篇论文最高分前 N 个 chunk
- [ ] 改为优先保留不同 section 的代表 chunk
- [ ] 避免保留多个相邻、重复、信息高度重合的 chunk

### 6. 增加日志与验证

- [ ] 记录每个 query 采用了哪种 chunk 保留策略
- [ ] 记录每篇论文最终保留了哪些 section 的 chunk
- [ ] 记录被截掉的 chunk 属于：
  - 重复
  - 噪声
  - 低价值 section
- [ ] 对比验证：
  - `paper_lookup` 的论文覆盖是否提升
  - 单篇论文总结类 query 是否退化

### 7. 改进回答过短的问题

- [ ] 将回答 prompt 从“只给简短结论”改为“先结论，再分析，再给证据”
- [ ] 为回答阶段增加可切换模式：
  - `简短模式`
  - `分析模式`
  - `深入模式`
- [ ] 在 `analysis / 深入` 模式下放宽“回答保持简洁”的提示词约束
- [ ] 提高回答阶段可使用的 evidence chunk 数量上限
- [ ] 对单论文理解类 query，允许回答阶段看到更完整的 section coverage
- [ ] 减少过强的 answer sanitize，避免把模型已生成的分析性内容删掉
- [ ] 为不同 query 类型定义不同回答模板：
  - 论文集合类：先总结整体，再列代表论文
  - 单论文理解类：先概述，再分方法/实验/结论展开
  - 比较类：先给结论，再分维度比较
- [ ] 在日志中记录：
  - 实际使用了哪种回答模式
  - 回答阶段使用了多少条 evidence
  - 回答长度是否被后处理明显压缩

## 一句话原则

- 找论文集合：限制单篇论文 chunk，占更多论文
- 读懂单篇论文：放宽 chunk 限制，优先做 section 覆盖
- 限制 chunk 时：先判断谁是关键 chunk，再做截断
- 回答生成：不要只压缩成一句结论，应按 query 类型决定展开程度
