# 框监督层级图学习与判别基线实现

日期：2026-09-18。提交者：chatgpt。

依据 `docs/algorithm/Alg_SupervisedGraphLearning.md` 实现静态图片的框监督学习、固定实体证据分类头和无框滑窗检测基线。本次仅进行 Python 静态编译及 Notebook 代码单元编译，不运行训练、推理、单元测试或调参；以下为实现说明，不是效果验收报告。

## 1. 文件与入口

| 文件 | 职责 |
| --- | --- |
| `src/nns/memorygraphs/supervised_data.py` | 监督配置、XML 标注、原图分组划分、ROI 变换、有效性/归属/软前景掩码 |
| `src/nns/memorygraphs/supervised_graph_learning.py` | 成员分组选择、相似区域角色对应、标注实体事务、重复观察 ledger、撤回与检查点 |
| `src/nns/memorygraphs/supervised_readout.py` | 固定 CNN 接口、类别证据读出、分类头训练、分类/多标签/检测指标、滑窗与 NMS |
| `src/nns/memorygraphs/supervised_experiment.py` | 数据集建图、冻结查询缓存、评价与 RSS/显存记录 |
| `src/test_supervised_graph_learning.ipynb` | 框图示、分组划分、建图、留出分类、可选分类头和无框检测入口 |

原 `graph_memorypool_onceoptimizer.py` 仅增加监督实体元数据、关联模态组权重、受保护实体处理及“只更新区域间关系、不生成无监督提案”的开关。无监督 API 的默认行为保留，监督流程单独调用 `SupervisedGraphOptimizer.learn_object(observation, features, search_budget=None)`。

## 2. 数据与坐标

XML 必须显式指定 `1based_inclusive` 或 `0based_halfopen`，不猜测端点约定。输入为 `[1,3,H,W]`、范围 `[0,1]` 的有限浮点 RGB。原图目标框带上下文等比例映射到固定画布，记录双向 3×3 仿射变换；图像采样使用像素中心、框坐标使用像素边界，框可逆映射回原图。

CNN 接收上下文图像后才用 ROI 限制区域构建，避免先清零框外而制造目标框边缘。valid mask、ROI、软前景概率和其他框重叠的归属不确定性分别保存。框内支持仅标为 bbox-only，不宣称为前景真值。方向继续使用既有置信度和弧度合同。

按原图划分 memory_build/readout_fit/validation/test，统一固定类别表；类别覆盖不足显式失败。演示选高频三类只为缩小入口，不是调参结论。无目标条目的 XML 不自动作为背景阴性监督。

## 3. 监督建图事务

1. 校验类别、来源、预处理版本和 observation ledger。重复的完整标注观察返回 `ALREADY_OBSERVED`；来源标签冲突或未改版本的标注变更返回 `INVALID_ANNOTATION`。
2. 在临时 GmemI/II 中构建框内区域。grad 与其 aps/ori 子区域作为整组参与选择；以质量、空间网格覆盖和支持规模排序，同时限制成员数、组数与展开叶槽位数。
3. 在冻结的原记忆中执行既有二进制粗筛及区域精验。出现区域歧义、不可评估的同类模板、预算中断或无效几何时不提交。
4. 同类 GmemIII 的角色对应不再要求新旧区域 ID 相等：用连续特征精验建立候选区域对，在平移假设下使用矩形 Hungarian 一对一分配，检查正向/反向覆盖及属性组一致性，再进行实体叶槽位和几何精验。实现不额外依赖 SciPy。
5. 复用兼容的同类实体，或者创建一个标注实例提案；更新 GmemII 区域间关系、成员/实体关系与独立 episode 证据。新候选的建立不宣称数学上证明了新颖性。
6. 在复制的池中完成全部修改后再替换活动池，并提交 ledger。成员更新后重新检查属性是否仍绑定 grad 以及组/叶预算；不满足时回滚整次观察。

标注种子直接进入 stable 检索池，但独立证据为一次。`confirmation=annotation_confirmed`、`protected`、`independent_support` 和 `visually_confirmed` 分开记录。实体保存标签、标注来源、成员弱前景权重以及相对 root 的规范化目标框。

同一组 grad/aps/ori 的成员权重按组内成员数分摊，避免简单把三个相关属性当作三个等量独立区域。现有 `_update_entity` 的重复证据、待确认成员、几何均值和成员淘汰机制继续用于兼容实体；新增/删除成员后必须通过监督分组预算审查。

可选 `SearchBudget(max_seconds=None, max_template_pairs=None)` 默认不限制搜索。它是协作式匹配预算，不是包含 CNN、构图和池复制的整步硬截止。耗尽返回 `UNRESOLVED`，不持久写入部分图。

## 4. 生命周期与检查点

普通候选淘汰不会删除受保护的监督实体。无监督实体更新与等价实体合并跳过监督实体，避免标签和 ledger 在无监督操作中失配；监督入口通过同类匹配完成观察复用。未实现监督实体之间的自动压缩合并。

`withdraw_entity` 显式删除整个错误实体并清理父索引和 ledger，保留可共享低层区域。部分标注证据的精确扣除需要从正确来源重建，不能仅减 support 就恢复几何统计。

检查点保存三层图、特征合同、监督配置、类别表、ledger、图版本、划分、随机种子配置及可选分类头/Adam 状态；恢复时检查合同及引用关系，允许切换设备。使用临时文件替换保存。图对象快照通过 `torch.load(..., weights_only=False)` 恢复，只接受本地可信检查点。保存的随机状态为来源记录，载入不自动覆盖调用者的全局随机流。

## 5. 分类与训练

每类对 accepted 实体的 `score × matched_coverage × exp(-error²/(2σ²))` 取最大值，避免模板较多的类别因简单求和天然占优。固定类别表对应固定 `6C` 维输入：分数、可评估覆盖、命中覆盖、归一化几何误差、缺失和搜索未完成标志。

支持固定证据阈值/间隔拒识；输出 top-k、实体 ID、root、恢复框、搜索状态和图版本。正匹配可形成预测，但稀疏搜索未命中返回 UNRESOLVED；还单列 `assignment_search_complete` 与 `absence_proven`，不把穷举平移等同于穷举所有叶指派。UNKNOWN 是当前搜索/判别算法下的拒识状态，不是“物体必定未见过”的证明。

可训练基线为 `Linear(6C,C)`，冻结 CNN 与图；单标签采用训练集类别加权交叉熵，多标签采用屏蔽未知标签的 BCE。Adam、批大小、轮数、早停等初值均置于 `SupervisedConfig`。仅 validation loss 选择最佳头与优化器快照。训练器检查 memory_build、readout_fit、validation 原图不相交；旧图版本的缓存/头不能用于更新后的图。

本实现采用方案允许的纯实体证据头，没有加入可选 128 维 GmemII 共享 MLP；无实体响应时的判别信息有限。少样本实验仍应将建图、训练头及调阈值的全部标签计入监督预算。

## 6. 无框检测及评价

`ProposalDetector.detect(image)` 不接收真值类别或框。枚举多尺度/宽高比滑窗，调用相同规范化编码与层级检索，通过模板相对 root 的目标框恢复原图坐标，再做类别内 IoU NMS。保留同模板多个位置的 accepted 实例，不将每类最大分类响应误当作全部检测结果。候选预算截断和检索未决分别输出。

分类指标包含所有失败样本的 accuracy、macro-F1、逐类召回、混淆矩阵，以及另列的接受覆盖率/准确率、未决率。多标签提供已知标签项上的 AP 与 macro/micro-F1。检测提供指定单个 IoU 阈值的全点插值 AP/mAP、proposal recall、重复和未匹配/定位错误统计；这不是 COCO 多阈值 mAP，也尚未细分所有误检成因。

当前检测为滑窗加模板恢复框的基线；文档中的可选框回归细化、GmemII 共享 MLP、跨模板压缩以及视频/真实光流跟踪仍属后续扩展。没有将 Retina 中的零光流占位输出用作前景分离证据。

## 7. 调试入口与检查范围

新 Notebook 的建图、查询、分类头训练和检测均有独立开关，默认关闭，执行结果为空。第 2 节给出原图框、规范化输入和有效 ROI 图示；第 4 节逐目标记录状态、独立支持、图规模、编码/总耗时、进程 RSS/峰值 RSS、CUDA allocated/reserved/peak，并逐行保存 JSONL。

静态编译范围：修改后的原图模块、新增四个 Python 模块，以及新 Notebook 的九个代码单元。未导入并执行这些模块，未运行数据集或 CUDA 测试，因此编译通过只能说明语法有效；实际运行兼容性、匹配/框恢复质量、内存和速度均待后续调试验证。
