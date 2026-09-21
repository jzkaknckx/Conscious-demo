# 长期训练诊断接口与高级模态 grad 门控分区修正

日期：2026-09-21。提交者：chatgpt。

算法报告见 [Alg_LongTermSupervisedTraining.md](../algorithm/Alg_LongTermSupervisedTraining.md)。报告包含可直接预览的美元符号 LaTeX 公式，区分已实现接口与后续候选选择、联合消歧、记忆压缩及增量事务方案。

本次代码变更：

- 图模块默认取消 aps/ori 直接继承 grad 分区。curv、aps、ori 首先通过 grad 局部轴向相干度门限，随后执行各自连续性分区。新模式的属性区域不带复制父 ID；aps 在该模式走 Trace_1D，避免面域强边缘屏障与前置边缘支持冲突。
- 不使用 grad 的最终标签、强度写入阈值和细化脊线代替前置门控。缺 grad、当前位置非有限或零幅值均不进入属性支持。阈值默认跟随 tau_grad_coh，使用严格大于号。
- 同步 FeatureResponseCache/query_l1 检索门控；分区合同升级，新旧检查点不能直接混用。
- learn_object 增加 profile_stages 接口和构图、选择、区域匹配、实体匹配、整池复制、提交阶段耗时，结果包含各模态支持/区域/入选统计与匹配工作计数。CUDA 同步计时只在显式开启时进行。
- SupervisedExperiment 以不可变的完整划分管理数据，max_images/start_image 只控制本次执行范围。批量扩大范围不会改变实验划分；旧截断划分检查点不强制迁移。
- 更新监督 Notebook：新算法配置、可选三模式对照、门控不变量、持久图引用检查、分段时间及模态数量表、完整划分上的连续窗口。用户手动点击运行，未自动启动训练。
- 旧绑定测试明确指定 legacy 开关；新增四个 grad 门控回归用例，供用户运行。

验证范围：对涉及 Python 文件和 Notebook 代码单元进行静态编译及 diff 空白检查；不运行训练、推理或调参。新算法效果、吞吐和回归行为仍待执行验证。特别是旧的全目标截断否决与区域歧义否决分支，本次没有声称已经解决。

静态编译结果：5 个 Python 文件、11 个 Notebook 代码单元通过，未执行回归用例。新 Notebook 清除旧执行输出，已将 Git HEAD 中含 7 个输出单元的基线副本保存至 `results/supervised_graph/notebook_baseline_from_HEAD_20260921.ipynb`；该副本明确是已跟踪版本，供查阅旧算法运行结果。运行新 Notebook 前需重启内核，避免继续使用 Python 已缓存的旧模块。
