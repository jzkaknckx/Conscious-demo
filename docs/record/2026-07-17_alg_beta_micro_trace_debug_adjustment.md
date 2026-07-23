# 2026-07-17 Alg_beta MICRO 追踪调试修正

- 在 `docs/algorithm/Alg_beta.md` 中补充 `BOUNDARY_SEEK` 长期滞留与 `CONTOUR_TRACE` 早退的原因分析。
- 将 `BOUNDARY_SEEK -> CONTOUR_TRACE` 的判据从全局 `drive.max()` 修正为实际落点的局部边界命中置信度与前向轮廓支持。
- 补充切向初始化、边界访问抑制延迟、迟滞阈值、最小追踪步数等状态机调整建议。
- 补充 `_extract_and_match_local_features()` 的拓扑上下文相似度调整建议，要求边缘原型复用与 GmemII 观察关系记录解耦。

