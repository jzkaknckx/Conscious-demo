# aps / ori 绑定 grad 分区

日期：2026-09-18。按用户要求将 aps、ori 定位为边缘的补充描述，采用共享 grad 分区，而非仅将独立分区参数调成相同数值。

## 实现

修改 `src/nns/memorygraphs/graph_memorypool_onceoptimizer.py`：默认 `once_bind_edge_attributes=True`。grad 仍通过现有门控、细化、亲和连接、连通分区及预算筛选形成区域；aps/ori 跳过自己的亲和连接和连通分区，改由 `EdgeAttributeRegionAssembler` 投影这些 grad 区域。

每个被保留的 grad 区域，对每种属性模态最多派生一个区域：

`attribute_pixels = grad_region.pixels ∩ attribute_writable_pixels`。

属性支持掩码为 grad 支持与自身可写门控的交集；属性沿用 grad 的质量、切向、事件和有效种子。原始 aps/ori 向量及各自通道 mask、门控仍保留。有效像素为空时跳过该属性区域，并记录 `edge_attribute_no_valid_pixels`；不为了与 grad 数量相同而写入无效值。

所有派生区域携带观察级 `parent_region_id`，指向本次 grad 区域，不是 Gmem 节点 ID。有效性空洞只删除不可写像素，不再次将区域拆成多个连通分量；记录 `edge_attribute_validity_holes`。邻接关系仅保留 grad 邻接中两端都有效的边，不跨空洞虚构局部连接。曲线排序无法完成时，原有采样器记录 `curve_order_unresolved`，不把它标为完整曲线。

区域支持维度继承 grad，含 grad 的事件区域。aps/ori 的可写描述子作为边缘补充采用 Trace_1D 拓扑提示。父 anchor 在子区域仍有效时复用其位置，否则由原采样器在有效点中选择；具体采样点和数量仍受属性有效性、接触端点和独立样本预算影响，不要求三模态逐点完全相同。

GmemI 特征节点、GmemII 星型和已有接触构建规则保留，各模态仍分别写入自己的原型。此次没有把属性向量拼进 grad 原型，也没有更改 Gpos 相似度或强行合并三种模态的记忆。

## 预算、缺失输入与对照

总区域预算仍为 256，调度按 grad 及其当前存在的属性组成的区域组预留份额，再分配其他独立模态，避免复制属性区域突破总上限。极小预算下优先保留 grad，属性可以被截断，并记录 `edge_attribute_region_budget`。

缺少 grad 时，aps/ori 不回退到独立分割：清空其可写/支持区域，输出警告及 `edge_attribute_missing_grad`。这使“边缘补充”语义在只有属性输入时仍一致。需要旧独立方式作对照时可显式设置 `once_bind_edge_attributes=False`。RGB 和 curv 继续原有独立分割。

记忆快照增加 `segmentation_contract`；加载时检查绑定/独立模式，不静默混用两种区域定义。旧快照未提供该字段时视为独立模式；绑定模式下应重新建图。

## Notebook

第 3 节 aps/ori 分区图按 `parent_region_id` 着色，与 grad 使用同一色标范围，便于对应同一段边缘。增加属性区域、父区域、像素数、样本数及异常原因表；导出区域记录包含父 ID。未改变参数以掩盖碎分。

图示仍使用有限色表，颜色可能重复；精确对应以父 ID 为准。由于缺失特征门控，属性图上可能出现与 grad 不完全重合的空洞；这些不代表新增分区。

## 验证

21 项小规模 CPU 回归通过，包含新增测试：

- 沿两段 grad 边缘交替改变 aps 外观，仍保留每父区域一个属性区域，且特征值没有被替换成 grad；
- 属性出现非有限值空洞时不新增区域，也不采样非法点；
- 完整支持时继承父邻接和 anchor；
- 缺少 grad 时不写入属性；总区域预算为 0/1/2/3/5 时不越界。

全部 Notebook 代码单元及模型静态编译通过。

在 `ILSVRC2014_train_00060030.JPEG` 上做 CPU 前向与一次构图对照，使用相同数值修复后的 CNN、相同阈值：

| 指标 | 独立 aps/ori 分区 | 绑定 grad 后 |
| --- | ---: | ---: |
| grad 区域数 | 52 | 52 |
| aps 支持像素 | 21,913 | 2,207 |
| aps 区域 / 节点 | 51 / 385 | 52 / 344 |
| ori 支持像素 | 48,493 | 2,203 |
| ori 区域 / 节点 | 51 / 282 | 52 / 343 |

本次每种属性的 52 个区域分别对应 52 个 grad 父区域。区域总数/节点数不保证减少：原先也受总预算限制，且覆盖对象不同；本次的目标是让属性不再独立划出大面积、碎片化分区。grad 本身的碎分会被继承，若它也不理想，需要继续审查 grad 连通性。

证据与脚本：[results/edge_binding_20260918](../../results/edge_binding_20260918/)。未运行完整长期训练，也未重新进行 CUDA 性能测量。重启 Notebook 内核，从头编码并重建记忆后再查看图示。
