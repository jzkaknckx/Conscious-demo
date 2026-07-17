# LEARN_MICRO Gpos 目标场修订记录

- 提交者：chatgpt
- 时间：2026-07-16 20:20:20 CST
- 类型：算法文档修订

## 更改概括

修订 `docs/algorithm/Alg_beta.md` 中 `LEARN_MICRO` 各拓扑意图的候选场与目标场定义：

1. 将 `SURFACE_CONFIRM`、`INTERIOR_SAMPLE`、`BOUNDARY_SEEK`、`CONTOUR_TRACE`、`RESUME_OR_EXIT` 的计算式改写为基于 GposI/GposII 响应的组合。
2. 明确新对象学习时主要依赖 GposI Surface/Color/Boundary/Corner 等响应，GposII 结构先验在实现可靠前置零。
3. 将距离、切向、步长与访问抑制项定位为眼动约束项，而不是独立候选来源。
4. 同步更新 MICRO 切换条件中的边界覆盖与内部残差判定符号。
