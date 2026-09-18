# VOC2007 监督学习 Notebook 与两图调试入口

提交者：chatgpt。日期：2026-09-18。

## 修改内容

修改 `src/test_supervised_graph_learning.ipynb`，数据根目录设为：

`/home/p/code/PASCAL_VOC2007/OpenDataLab___PASCAL_VOC2007/raw/VOCdevkit/VOC2007`

读取 VOC XML 时使用 1-based inclusive 坐标合同，默认类别为 cat、dog、car；不修改 CNN、相似度或分割阈值。默认排除 difficult 标注目标，属于自定义框分类协议，检测指标仍不能等同 VOC 官方 difficult-aware mAP。

官方 train 按固定种子拆为 75% memory_build 和 25% readout_fit，官方 val 保留为 validation。官方 test 未解压时保留空 test，不用 validation 替代并标记成 test，也不把后续解压的测试图片纳入训练。检查原图交叉、缺失图片/标注及非空集合的类别覆盖。

## 两张单图演示

从 memory_build 中确定性选取同类的两张不同原图，每图取该类面积最大的可用目标。两图共用演示记忆，分开执行以下过程：

- 原图目标框、其他对象框、规范化 CNN 视图与有效 ROI 图示，附坐标往返值。
- 五模态通道形状、有限值比例、NaN/Inf 数量及数值范围。
- 学习前检索，观察第二张图片对已有实体的跨图激活。
- 实际构图报告中的五模态分区、入选锚点、aps/ori 父区域 ID、区域采样数和截断/拒绝原因。
- 学习状态、三层节点增量、实体成员相对 root 的布局、独立支持与标注/视觉确认状态。
- 学习后原图查询的类别分数、实体位置、覆盖率、几何误差与搜索完整性。
- 成功提交后重放同一标注，检查节点数和独立支持是否保持不变；失败观察不自动重跑完整匹配。
- 单独记录编码/学习耗时和内存资源。学习后原图检索只作为自检，不计入泛化指标。

为避免调试图额外运行一次分割，在 `SupervisedGraphOptimizer.learn_object` 增加默认关闭的 `capture_debug=False` 参数。开启时，`last_debug` 保存本次实际 BuildReport、入选区域 ID、模态组映射和选择诊断；只保留最近一次观察，不写入检查点。下一次学习会清空，Notebook 展示完成后也主动释放。

## 批量入口

`RUN_SINGLE_DEMOS=True` 默认开启两图示例；`RUN_MEMORY_BUILD=False`，由用户完成两图检查后开启。批量默认处理 20 张建图原图中的全部目标，`BATCH_IMAGE_LIMIT=None` 可扩展至整个建图集。留出查询单独受 `QUERY_IMAGE_LIMIT` 控制；小样本类别不齐时分类头会拒绝拟合，需扩大范围。

批量开始时释放演示记忆，独立初始化图，避免并存占用及演示证据混入统计。批量是顺序数据集学习，不是并行图事务。不限制单步搜索时间，保留 SearchBudget 配置入口。

逐目标打印并写入 JSONL：状态、失败原因、实体 ID、支持次数、三层节点数、编码/总耗时、RSS、峰值 RSS、CUDA allocated/reserved/peak。绘制耗时、图规模、内存曲线。默认每 10 个目标保存检查点，结束和 KeyboardInterrupt 时保存最后完整提交的图。再次运行批量块会沿用当前 learner，通过 ledger 去重；也提供 `RESUME_CHECKPOINT` 加载入口，要求配置与划分一致。

保留冻结查询、分类头和无框检测入口。官方 test 缺失时明确跳过测试评价；检测示例可以使用 validation，但明确标记其来源，不输出伪造测试成绩。

## 验证范围

四个监督 Python 模块和新 Notebook 的十个代码单元静态编译通过；`git diff --check` 通过。未执行模型、两图推理、训练或调参。原 `src/test_memorypool_onceoptimizer.ipynb` 的已有改动和运行记录未覆盖。
