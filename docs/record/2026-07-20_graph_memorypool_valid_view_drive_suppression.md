# graph_memorypool 有效视野 drive 抑制记录

- 提交者：chatgpt
- 时间：2026-07-20 15:58:16 CST
- 类型：bugfix

## 更改概括

针对输入图像小于配置视野时外侧零填充产生黑边伪边缘、干扰注视点选择的问题，对 `src/nns/memorygraphs/graph_memorypool.py` 进行修正：

1. 在 `InterestOptimizer` 中新增统一的有效视野边界、有效视野 mask 与 `suppress_invalid_view()` 方法。
2. 将 `initialize_base_interest()` 中原有手写黑边压制替换为统一抑制方法。
3. 将 REVIEW、LEARN_MACRO 与 LEARN_MICRO 各意图最终用于选点的 `drive` 接入有效视野抑制。
4. 修正最终 drive 包含距离惩罚时的黑边处理：无效区域填入极小值，避免有效区 drive 为负时黑边零值反而被选中。
5. 在 MICRO 的兜底随机选点中只从有效视野内采样。

## 验证

- 已运行 `python3 -B -m py_compile src/nns/memorygraphs/graph_memorypool.py`。
- 已运行 `git diff --check -- src/nns/memorygraphs/graph_memorypool.py`。
