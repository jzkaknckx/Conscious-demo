# 三层超图检索、记忆巩固与 Notebook 调试实现

提交者：chatgpt。记录时间：2026-09-12 15:48:12 CST。

## 更改概括

按照 [Alg_GraphandOptimizer.md](../algorithm/Alg_GraphandOptimizer.md) 的首阶段路线修改 [graph_memorypool_onceoptimizer.py](../../src/nns/memorygraphs/graph_memorypool_onceoptimizer.py)，并更新 [test_memorypool_onceoptimizer.ipynb](../../src/test_memorypool_onceoptimizer.ipynb)。本次不修改算法文档、CNN 或用户行为规范，不进行模型运行、调参或性能实验。

## 代码实现

- 为区域、实体、从属关系及关系边增加持久证据统计，区分原始命中、独立 episode 支持、验证机会、几何均值与方差。
- 增加 GmemII 区域类型、版本、区域间关系及接触端点证据索引；GmemIII 增加根槽位、可重复成员角色、从属反向索引、布局关系和候选状态。
- 用 `region_view` 保留源目标及循环约束，拒绝无法解析的折叠实例或冲突布局；以 `entity_view` 按成员分组虚拟挂载到统一锚点。
- 替换旧 GposII 平滑占位路径，返回固定姿态下全图结构响应、验证峰、覆盖率、角色位图、赋值及几何残差；新增 GposIII 查询接口。
- 增加按现有分组度量计算的底层候选、共享描述子类别、静态成员位图、反向关联投票、动态角色位图和 LRU 响应缓存。缺失模态、未评估与已评估失败分开处理。
- 增加 `GraphConsolidationOptimizer`：查询已有记忆，临时构图，按区域做局部平移对齐，再暂存并提交长期更新；稀疏未命中不会直接被当作新结构，而会经过局部密集补查。
- 生成有界、允许重叠的空间／接触组合，匹配实体角色，更新从属与布局证据；反复出现的新成员先挂入待确认集合，可靠缺失成员可降权／移出。
- 等价区域及实体仅在带属性几何一致时合并，并合并 episode 来源而非累加重复支持；不相容布局保留为独立候选。低支持实体候选按预算与未出现时长回收，稳定实体不按该规则删除。
- 增加版本化 `state_dict/load_state_dict`、显式实体构造夹具及独立的只读层级查询。持久化同时包含特征契约，旧池不自动冒充新格式。

## 接口说明

- `Controller.run_step(..., learn=True)` 和 `handle_new_view(..., learn=True)`：传入 GmemIII 时执行完整学习，返回 `LearningResult`。
- `learn=False`：保留原始一次性观察写入和 `BuildReport`，适合调试分割与采样。
- `learn_view(features, source_id=None, episode_id=None, valid_mask=None)`：长期学习入口；未给 episode 时使用输入内容摘要，防止重复输入被当作独立观察。
- `query_hierarchy(..., exact=False)`：默认稀疏候选与位姿查询；`exact=True` 遍历全部整数平移位置，适合小图对照。
- `l2_structure_synthesis(..., gmem_i=..., valid_mask=...)`：额外传入 GmemI 可正确剥离旧 GposI 的已知模态权重；保留旧的 x/y/score 等返回键，并增加 response/matches/diagnostics。
- `LearningResult.observation_i/observation_ii` 与长期池分离；观察编号到长期编号通过 `region_mapping` 查询，不能混用。

## Notebook

- 保留原始分区／采样／边检查，明确使用 `learn=False`。
- 增加正式 GposII 与参考评分并排显示，标注预测峰、预期锚点和连续分数色条。
- 增加同 episode 重复学习示例、长期节点数量变化、原始命中／有效独立支持柱图及成员统计表。
- 增加 GmemIII 根角色／成员图、按成员着色的虚拟叶点图与多归属列表。
- 增加只读层级查询、位图、候选预算、倒排访问、覆盖率、几何残差及 GposII/GposIII 定位图；记录写入位置只用于绘制诊断标记，不传入检索。
- 保存结果时可同时保存版本化长期池。参数对照默认关闭；全部单元输出和执行计数已清空，避免旧结果误指向新实现。

## 当前实现边界

- 首版只实现固定尺度、固定方向的平移；没有新增旋转／尺度特征变换、独立部件形变或遮挡分类器。
- 底层使用原度量的精确比较及采样事件，没有引入外部 ANN 依赖。稀疏查询具有采样、位姿桶和候选预算近似，明确返回 `search_complete=False`。
- 密集模式覆盖全部整数平移位置，但局部赋值采用确定性最大响应与冲突拒绝，不执行一般图组合回溯；`assignment_search_complete=False` 单独说明该边界。
- 实体组合与巩固是文档允许的有界启发式方案，未实现全局子图解释损失优化。匹配多解时保守暂缓学习；没有把模型输出当作已验证的语义物体标签。
- 学习的局部密集回退和事务暂存可能占用较多时间／内存；实际效果、默认参数和耗时均待用户调试。

## 验证

- `python3 -m py_compile src/nns/memorygraphs/graph_memorypool_onceoptimizer.py`。
- Notebook JSON 解析与每个代码单元的 `compile(..., 'exec')`。
- 静态检查新增 `graph_*` 配置引用均有定义，检查重复方法名、本地文档链接及 `git diff --check`。
- 没有执行源码导入、CNN 前向、学习、检索或任何 Notebook 单元；编译通过不代表运行行为已经验证。
