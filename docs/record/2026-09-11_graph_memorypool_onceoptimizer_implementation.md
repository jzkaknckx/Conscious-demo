# 一次观察区域构图器与调试 Notebook

提交者：chatgpt。记录时间：2026-09-11 20:27:36 CST。

## 更改概括

根据 [Alg_onceoptimizer_beta.md](../algorithm/Alg_onceoptimizer_beta.md)，续写用户已准备的 [graph_memorypool_onceoptimizer.py](../../src/nns/memorygraphs/graph_memorypool_onceoptimizer.py)，并填充 [test_memorypool_onceoptimizer.ipynb](../../src/test_memorypool_onceoptimizer.ipynb)。

## 实现内容

- 在现有 `MemoryConfig` 中增加 `once_*` 参数；原有 Gmem/Gpos、相似度引擎及节点／边数据类保持不变。
- 新增具名 CNN 特征与整数子空间输入适配，支持显式有效视野、缺失模态、有效通道检查及非有限数值排除。
- 缓存整图门控与描述子信息，计算可靠度、方向、梯度法向非极大值抑制与平台种子；用同模态局部亲和边和带属性范围约束的并查集合并形成区域。
- 用局部出口数分离分支事件，选择实际采样锚点；按区域路径距离采样边界、孔洞附近及内部覆盖点，对可排序曲线保留端点和折线误差约束。
- 通过局部标签搜索形成稀疏区域接触，优先预留锚点和接触端点预算。区域预算为后续模态保留份额；不足处显式报告，不宣称采样完成。
- 默认按采样实例注册 GmemI 节点，区域星型与跨区接触分别写入 GmemII；提供原型复用对照。写入先在临时池中准备，再提交，避免描述子或几何计算失败污染原记忆。
- 保留 `Controller.run_step(X_subspaces, gmem_i, gmem_ii, gmem_iii, gpos, ...)` 接口；每次调用追加一次完整观察并返回 `BuildReport`。不执行旧眼跳状态机，不推进神经元疲劳时钟，也不自动写 GmemIII。
- 增加协调器 `handle_new_view()`、只读 `query_l1()`、写入几何核验，以及独立标注的固定姿态参考星型评分；后者不修改或冒充 GposII。

## Notebook

包含 20 个单元，其中 10 个代码单元：环境初始化、输入／配置、单次构图、各模态分区、单区域采样、星型／接触图、几何核验、只读平移与干扰图查询、可选参数对照、可选结果保存。

默认使用仓库内图片，支持切换合成图；首轮启用梯度与 RGB，额外特征可选。每次构图重新初始化记忆池。参数对照和文件保存默认关闭，所有执行计数及输出保持空白。

## 当前边界

- 构图与采样的默认参数尚未调试；复杂线域若不能可靠排序，将报告 `curve_order_unresolved`，保持未完成，不猜测拓扑。
- 有效区域按当前 Retina 裁剪坐标处理；其他预处理流程应显式传入对应 `valid_mask`。
- 首版不跨无支持间隙插补，不扩展 CNN 特征族；保留原有写入门控。
- GposII 原有位置约束缺失仍存在；参考评分仅用于之后独立定位写入与检索问题。

## 验证

仅进行 Python 编译、Notebook JSON／代码单元编译及文档链接检查，未执行构图、CNN 前向或调参实验。使用修改前后 AST 比较确认现有 Gmem/Gpos、相似度引擎与数据类无行为修改。
