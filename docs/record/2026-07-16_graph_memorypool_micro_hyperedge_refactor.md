# graph_memorypool MICRO 与关系边重构记录

- 提交者：chatgpt
- 时间：2026-07-16 21:16:05 CST
- 类型：代码重构

## 更改概括

依据 `docs/algorithm/Alg_beta.md` 中“眼动引导机制与图结构边关系的重构”草案，对 `src/nns/memorygraphs/graph_memorypool.py` 进行重构：

1. 新增 `MicroIntention`，将 `LEARN_MICRO` 内部拆分为 `SURFACE_CONFIRM`、`INTERIOR_SAMPLE`、`BOUNDARY_SEEK`、`CONTOUR_TRACE`、`RESUME_OR_EXIT` 等独立意图。
2. 将各 MICRO 意图的目标场计算封装为独立方法，`run_step()` 缩减为单步编排流程。
3. 新增 `RelationEdge` 多重关系边结构，支持同一端点对下的多位置关系、自指延展边与边界/轮廓边。
4. 扩展 `SemanticNode` 为“子节点集合 + 关系边集合”的超边式结构，同时保留 `peripheral_links` 作为旧接口兼容视图。
5. 调整 REVIEW、MACRO、能量聚合和 GposII 雏形逻辑，使其读取新关系边集合。

## 验证

- 已运行 `python3 -B -m py_compile src/nns/memorygraphs/graph_memorypool.py`。
- 运行端到端 smoke test 时环境缺少 `torch`，无法执行动态测试。
