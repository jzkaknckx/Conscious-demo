# 2026-07-20 Alg_beta 渐变边界与语义前沿学习探讨

- 在 `docs/algorithm/Alg_beta.md` 结尾追加渐变边界、语义前沿蚕食与 MICRO 意图扩展探讨。
- 分析 semantic 蚕食、多尺度梯度、动态阈值三类方案的优缺点。
- 提出连续性前沿学习 (CFL)：用 RGB 成员度场、语义前沿、累计颜色漂移和动态阈值识别宽渐变边界。
- 补充 `FRONTIER_ADVANCE`、`TRANSITION_PROBE`、`BAND_TRACE` 等 MICRO 意图建议，以及有效图像掩码 / padding 抑制与 `truncated_by_frame` 退出逻辑。

