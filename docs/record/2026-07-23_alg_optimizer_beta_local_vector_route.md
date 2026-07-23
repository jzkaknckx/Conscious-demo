# 2026-07-23 Alg_optimizer_beta 视窗口向量引导路线

- 新建 `docs/algorithm/Alg_optimizer_beta.md`，作为视窗口向量引导优化器的技术商讨文档。
- 明确新路线核心：MICRO 从旧的全局 `drive` 选点改为由视窗口内主要特征流形合成眼动向量。
- 补充基于动态竞争的 MACRO/MICRO/REVIEW 状态机，加入 GposII/GposIII 视窗口外强响应的 REVIEW 抢占与挂起恢复机制。
- 设计 `H*W` 动态引导矩阵，包括一维边缘蝴蝶状响应、二维 surface frontier 响应、点事件和关系缺口响应。
- 补充视窗口特征写入流程，包括关键点采样、GmemI 复用/创建、GmemII 开启、填充、闭合、挂起与恢复。
