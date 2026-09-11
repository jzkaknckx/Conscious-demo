# graph_memorypool_newoptimizer 视窗口向量优化器实现记录

提交者：chatgpt  
提交时间：2026-09-07 22:24:34 CST

## 更改概括

根据 `docs/algorithm/Alg_optimizer_beta.md`，在 `src/nns/memorygraphs/graph_memorypool_newoptimizer.py` 中实现新的视窗口向量引导优化器主路径。

## 主要内容

1. 新增动态工作视窗口、固定注意力预算、状态竞争观测量与调试缓存。
2. 将 MICRO 下一眼从全图 `argmax drive` 改为局部势场一阶矩向量：动态矩阵由边缘蝴蝶核、surface 外沿壳层、点事件、关系缺口和访问抑制共同更新。
3. 新增 `MACRO_SEARCH`、`MICRO_BUILD`、`REVIEW_VERIFY`、`SUSPEND_RESUME` 行为状态，并把各状态处理封装为独立方法，缩短 `run_step()`。
4. 在视窗口内采样关键点并即时写入 GmemI/GmemII，包含 anchor 关系、时间邻接关系和二维 surface 自指外延关系。
5. 保留 `debug_optimizer`、`matrix1` 到 `matrix4` 等必要调试接口，便于观察窗口分数、状态分数和局部向量组成。

## 检查

已执行：

```bash
python3 -m py_compile src/nns/memorygraphs/graph_memorypool_newoptimizer.py
git diff --check -- src/nns/memorygraphs/graph_memorypool_newoptimizer.py
```

当前环境缺少 `torch`，未执行真实前向冒烟测试。
