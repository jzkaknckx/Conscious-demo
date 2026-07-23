# graph_memorypool 边界追踪迟滞修正记录

- 提交者：chatgpt
- 时间：2026-07-17 22:14:01 CST
- 类型：代码修正

## 更改概括

依据 `docs/algorithm/Alg_beta.md` 第 6.1 与 6.2 小节，对 `src/nns/memorygraphs/graph_memorypool.py` 的 MICRO 边界学习状态机进行修正：

1. 移除 `BOUNDARY_SEEK` 中基于全局 `drive.max()` 的即时 `CONTOUR_TRACE` 切换。
2. 新增 `pending_boundary_target`，使 `BOUNDARY_SEEK` 只选择候选点；下一步实际落点完成局部观察后，再用边界命中置信度与前向轮廓支持判断是否进入 `CONTOUR_TRACE`。
3. 为 `CONTOUR_TRACE` 增加进入/退出迟滞阈值、最小追踪步数 `trace_min_age`，避免一两步早退。
4. 将切向初始化改为基于局部边界几何估计，避免使用从面域内部跳向边界的逃逸眼跳向量。
5. 将边界覆盖访问图与目标选择抑制图拆分，并延迟到目标选择后再标记追踪进度。
6. 增加角点、交点、端点等局部角色响应保护，避免把拓扑事件点直接视为普通追踪失败。

本次按用户要求未修改 `_extract_and_match_local_features()`。

## 验证

- 已运行 `python3 -B -m py_compile src/nns/memorygraphs/graph_memorypool.py`。
- 已运行 `git diff --check -- src/nns/memorygraphs/graph_memorypool.py`。
