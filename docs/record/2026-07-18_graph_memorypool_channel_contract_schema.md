# graph_memorypool 通道契约与分组相似度实现记录

- 提交者：chatgpt
- 时间：2026-07-18 22:06:15 CST
- 类型：代码重构

## 更改概括

依据 `docs/algorithm/Alg_beta.md` 最新章节“补充：探讨二修订版 - 多模态通道依赖、门控与相似度统一规范”，对 `src/nns/memorygraphs/graph_memorypool.py` 进行最小落地实现：

1. 新增 `feature_specs` 通道契约，显式区分 `gate_only`、`context_only`、`sim_only` 与 `gate_and_sim`。
2. 新增 `FeatureDescriptor`，将局部观察拆分为 `sim_mask`、`gate_mask`、`context_mask`、`gate_context` 与 `topology_context`。
3. 将 GmemI 相似度改为按 metric group 分组计算，并使用加权几何平均合成。
4. 将 `grad` 相似度从原始点积改为 `[gradx, grady]` 的归一化轴向向量相似度，`abs_strength` 与 `theta` 只进入门控和上下文。
5. 在 GposI 投影中乘回同一模态门控场，确保响应只出现在物理有效区域。
6. 在 `MultilevelCoordinator` 输入入口将旧 `grad` 规范化为 `[abs_strength, theta, gradx, grady]`，并兼容 RGB 与旧 `hue` 面域输入。

## 验证

- 已运行 `python3 -B -m py_compile src/nns/memorygraphs/graph_memorypool.py`。
