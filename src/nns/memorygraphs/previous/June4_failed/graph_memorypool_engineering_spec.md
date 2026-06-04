# Memory Graphs 演进算法讨论及重构建议

针对系统中出现的“Explorer -> Inspect -> Consolidate”无限死循环，以及 GmemII 语义拓扑生长停滞（节点数量保持为1）的现象，其本质原因在于：**拓扑结构的发展过度依赖硬编码的状态机逻辑，缺少时间维度的记忆衰减与基于数据的自底向上生长机制。**

现整合您的修正建议，提出如下完整的算法重建方案。并在必要处给出严谨的数学或逻辑约束表述。

---

## 1. 引入时间维度：Gmem节点的动态激活层 (Activation Dynamics)
原有逻辑中节点一经建立和匹配即视为绝对存在。这会产生大量冗余匹配并严重拖慢检索。我们需要引入一种随时间衰减的生物学记忆机制，只有足够“活跃”的节点才会参与当前状态的决断。

- **节点激活状态定义**
  对每一个节点 $n \in (\text{Gmem}^I \cup \text{Gmem}^{II})$，为其增加随时间演化的连续激活值 $A_n(t) \in [0, 1]$。
- **状态更新动力学**
  在每一时间步（例如一个运行周期 $t$）内，每个节点的激活量先依据固定速率产生指数衰减，再接受后续自底向上（如匹配到新特征）或自顶向下（如被正确预期）的激励增量补充，但最大值截断为1：
  $$A_n(t) = \min\left(1.0, \lambda_{\text{decay}} \cdot A_n(t-1) + \Delta A_{\text{bottom-up}}(t) + \Delta A_{\text{top-down}}(t)\right)$$
  其中 $\lambda_{\text{decay}} \in (0,1)$ 是遗忘率常数。
- **Gpos 检索的硬过滤**
  重构 `Gpos` 的匹配和寻址机制。为了提升速度和抑制噪声，无论是在生成路由图或是进行结构合成时，检索过程只看当前活跃度高于某个过滤阈值的子集：
  $$\text{FilterActive}(G) = \{ n \in G \mid A_n(t) \ge \tau_{\text{active}} \}$$
  这自然使得陈旧的背景噪音特征从全局图中“隐退”。

## 2. 脱离状态机的生硬约束：GmemI 的共激活辅助分割 GmemII
原模型中只有当状态进入 `INSPECT` 并满足条件时，才会被迫“冷启动”向 GmemII 中塞入新节点。应当改为时间域上的 **共激活学习（Hebbian Learning）**：

- **自下而上的拓扑聚合倾向**
  在一定时间步宽度的滑窗 $T$ 内，若两个 GmemI 上的分离特征（孤立节点） $n_i, n_j$ 往往能在相近的时间域被同时激发出高 $A(t)$。这种强烈的“时序相关”反映了它们大概率属于同一个现实物体。
- **动态 GmemII 生成**
  维护一个 GmemI 节点间的相关度矩阵 $W$。当共激活度 $\Delta w_{i,j} = \eta \cdot A_i(t) \cdot A_j(t)$ 不断累加超过建构阈值且，当前这两者无法被任何一个已知的 GmemII 连结解释时，系统会自动将这组高相关性的局部特征囊括起来，**突变生成（Genesis）** 一个新的 GmemII 语义锚点节点并挂载。
  这将解决 GmemII 只有一个节点的僵局，它是基于数据“浮现”的，而非由状态机的单一分支强制产生的。

## 3. 融合空域视角的物理连通先验（单连通分量辅助）
如果仅仅看过去时间内的兴趣与注意力的拼凑，依然有“缝合怪物”的风险（例如，从杯子直接跳到桌面产生了错误的拓扑）。为了将 GmemI 特征更好地归属于不同的 GmemII，我们利用视觉底层的连续分布作为空间约束。

- **分割引导（Affinity Mask）**
  使底层系统（例如 CNN、SWT模块或更底层的注意力机制）提供一个粗粒度的像素亲和度函数 $M_{\text{spatial}}(\mathbf{x}_1, \mathbf{x}_2) \in [0, 1]$，用以表示空间两点由于颜色/粗糙度/边界封闭性而在直觉上有多大可能性同属一个连通块。
- **惩罚非连通跨越**
  在评价一个注视位置提取提取特征是不是某个已知语义物体（GmemII）的一部分时，我们为该关联引入空间结构增益：
  $$\text{LinkEnergy}(n_{I}, n_{II}) = \alpha \cdot \text{Sim}(n_I, n_{II}) + \beta \cdot M_{\text{spatial}}(\mathbf{x}_{n_I}, \mathbf{x}_{\text{anchor}})$$
  如果这步眼跳跃过了一个清晰的边界，从而 $M_{\text{spatial}} \approx 0$，则极大地削弱此时将其纳入当前 GmemII 拓扑的可能。这使得算法拥有自我意识，在边界处主动断开链接并为新区域萌生下一个 GmemII 对象。

## 4. 削弱由于 SaccadeState 产生的强制决策权
如您所述，这非常关键：不能指望 `SaccadeState` 的生硬切换本身去决定世界模型该长成什么样子。状态和记忆必须解耦：

- `Gmem` (I 及 II) 是一套独立于注视的演化后台引擎——无论眼动在哪里，只要看到新特征、旧特被匹配，网络就依据上述 1、2、3 条自发进行着强化（巩固结构）与惩罚（新建或隔离结构）操作。
- `SaccadeState` 退回到唯一功能：**修改与生产兴趣图**并**提供下一步最佳注视坐标**。状态机变成了基于当前 `Gmem` 生成信息的注意力导演（Director），它的决策不再是对记忆进行 CRUD，而是单纯寻找“最具价值的查看视角”。

## 5. 重构具备自省能力的状态决策器 (Intelligent Saccade Switcher)
为了打破僵硬的 `EXPLORE -> INSPECT -> CONSOLIDATE` 三态循环，需要将其升级为基于信息量与概率判断的动态图。定义效用评估：
* $A^{*}_{sem}(t) = \max_{k \in \text{Gmem}^{II}}(A_k(t))$：全局最高语义激活值，代表“我有多大把握现在正盯着一个已知物体”。
* $I_{loc} = \int_{N(\text{fixation\_point})} \text{interest\_map}(\mathbf{x}) d\mathbf{x}$：局部未解释兴趣信息的残余量。

**新的智能转移函数**：

1. **状态：EXPLORE (全局遍历)**
   - **行为**：全局跟随长周期兴趣图抑制残余热点。
   - **退出机制**：若某次眼跳后引发局部的 GmemI 发生了聚集高激活，或者发现了巨大的局部新鲜度 $I_{loc}$ ，此时强制跳转入 `INSPECT` 以验证细节。
2. **状态：INSPECT (特征聚拢提取)**
   - **行为**：沿着特征连通连通先验 $M_{\text{spatial}}$ 在原注视点周边作密集的局部微眼跳。
   - **退出机制**：若汇集的特征足以推高了某一个 GmemII 的激活值满足 $A_k(t) > \tau_{recognize}$ ，说明“认出物体部件了”，平滑切入 `CONSOLIDATE` 期望进行拓扑预测；反之，如果在该连通区域 $I_{loc}$ 耗尽，仍然无法推高任何 GmemII （这代表当前面对彻底的新事物），则切回 `EXPLORE` 寻找别的区域（此时后台第2点的共激活聚合通常已经自动建好了一个新的 GmemII，无需显式干预）。
3. **状态：CONSOLIDATE (图先验校验)**
   - **行为**：此时状态机已经不再盲动，而是由当前最高激活的 $A^{*}_{sem}$ 释出期望的外围特征相对坐标进行自顶向下制导查询。
   - **退出机制**：期望的坐标接连数次校验落空导致图传唤能量枯竭，使得 $A^{*}_{sem}$ 指数滑落至无法维持“认知置信度”退坡回 `EXPLORE` 止损；或者预期全中，物体在短期内对视觉刺激达成了饱和，形成当前图结构块强烈的返航抑制（Inhibition Of Return），随即自然跃迁到 `EXPLORE` 对其余画面未知进行扫描。


# graph_memorypool.py 算法重构技术规格书 (Engineering Specification)

## 0. 概述 (Overview)
本文档针对 `/src/nns/memorygraphs/graph_memorypool.py` 中存在的“状态机无限死循环”以及“GmemII 层级语义拓扑停止演化（节点数始终为1）”的严重逻辑缺陷，提出代码实现层面的重构指南。技术团队需要严格按照以下五个维度的规格描述对现存代码结构进行改造。

核心思想：**将由状态机支配的生硬拓扑生成过程，替换为基于“记忆时域激活、局部空间连通约束规律及自底向上共同激活规律”的自发图生长的过程**。

---

## 1. 引入时间维度：Gmem 节点的激活状态与 Gpos 检索截断

为突破过往静态记忆的检索冗余堆积，为图中的节点灌注时间生命力。

- **[修改点 1.1: 节点数据结构]**
  - **位置**: `ModalityNode` 与 `SemanticNode` 的定义（通常位于对应 `Gmemory` 的内部或是顶端的数据类声明中）。
  - **变更**: 增设 `self.activation_level = 0.0` ($A_n(t)$) 和 `self.last_update_step = 0` 属性。
  - **方法新增**: `update_activation(step: int, gain: float, decay_rate: float)`，实现基于当前运行时间的指数衰减和刺激增量截断：$A(t) = \min(1.0, A(t) \times \text{decay}^{\Delta t} + \text{gain})$。
  
- **[修改点 1.2: 动态特征池映射机制]**
  - **位置**: `Gposition` 模块或相关 L1/L2 检索器的方法（如路由选择函数的入参准备处）。
  - **变更**: Gpos 执行 `l1_modality_routing_projection` 时，传递给检索池的必须是一个 **Active View**，即硬过滤去除 $A_n(t) < \tau_{\text{active}}$ 的节点。放弃全集检索，削弱过期的背景与游离噪音的干扰。

---

## 2. 脱离状态机硬编码：共激活辅助分割 GmemII (Hebbian Learning)

解决由于 `GmemII` 一直等待代码调用导致其停止生长的困境，通过时间维度的特征时序并发来生成语义。

- **[修改点 2.1: 新增共激活矩阵管理]**
  - **位置**: `GmemoryII` 的初始化与数据结构中。
  - **变更**: 新设 `self.co_activation_matrix = defaultdict(float)` 以记录孤立特征对 $(N_i, N_j)$ 双边共激活历史权重累计。

- **[修改点 2.2: 自发的拓扑绑定机制 (Genesis)]**
  - **位置**: `Controller.run_step` 中 `Bottom-Up` 特征聚合模块后的后台过程。
  - **变更**: 在抽取完当前步的 `active_gmem1_nodes` 后：
    1. 两两增加其共激活权重 $\Delta w_{i,j} = \eta \cdot A_i(t) \cdot A_j(t)$ 制约。
    2. 当检测到某配对的累计 $W_{i,j} > W_{\text{genesis\_threshold}}$ 且两者均没有绑定任何处于高活性的 `GmemII` Semantic节点时，**自发**且静默地触发生形逻辑，执行 `gmem_ii.add_semantic_node()`，而不是等待状态机的指令。

---

## 3. 增强空域先验视角（单连通组件辅助）

阻止网络将不在一个物理连续表面的独立特征盲目地“缝合”挂载进同一个语义拓扑中。

- **[修改点 3.1: 将亲和力引入能量函数 (Link Energy)]**
  - **位置**: `GmemoryII` 节点的 `add_peripheral` 和 `distortion_correction_and_branch` 方法。
  - **变更**: 改变原有单纯依靠位置距离计算的链接损失。接受一个来自底图中抽取的遮罩或空间相似度评价结果：$M_{\text{spatial}}(\mathbf{x}_{anchor}, \mathbf{x}_{prob})$。
  - **控制逻辑**: 为结构绑定添加刚性拦截。若跨越断裂地带（例如 $M_{\text{spatial}} \approx 0$ 的非连通组件跳跃），直接截断将其挂载为当前 `SemanticNode` Peripheral 的执行流（惩罚常数 $\beta \to \infty$）。由此强制 `Gmem` 对断裂地带的数据另起炉灶建立新的 `GmemII` 分支对象结构。

---

## 4. 权力下放：彻底剥夺状态机对图结构的增删越权决断

状态机制必须退回它仅有的使命：决定往图里哪一块投入注意力。它应该是个只读环境以及书写 `interest_map` 的角色。

- **[修改点 4.1: 拆解 Controller.run_step 内僵硬的分支执行]**
  - **位置**: `Controller.run_step`
  - **变更**: 删除类似于：
    ```python
    if self.state == SaccadeState.EXPLORE:
       # 如果找不到，强制 new_sem_node = gmem_ii.add_semantic_node(...)
    ```
    这类强制建图的代码。
  - **替代方案**: 将所有的 `add_semantic_node`, `add_peripheral` 等拓扑建立过程平移剥离出 `SaccadeState` 判断分支外，化为独立的后台运行模块 `_background_memory_consolidation()`，根据 2 与 3 条的规律自行演化。

---

## 5. 状态切换器升级：引入计算效用转移模型 

状态转换应基于环境量而非死循环，打断过拟合和无限空跳的窘境。

- **[修改点 5.1: 信息量观测标量的实现]**
  - **位置**: `Controller.run_step` 每帧处理前段。
  - **变更**: 必须计算出两个核心标量用于转移判断：
    1. 局部注视未解决残余兴趣度 $I_{loc}$（以固定注视窗为半径积分 `interest_map` 内的值）。
    2. 全局最大的对象认知把握 $A^{*}_{sem}(t) = \max_{k}(\text{SemNode}_k.\text{activation_level})$。

- **[修改点 5.2: 依据效能指标覆写状态路由 (Saccade Switcher)]**
  - **位置**: `Controller.run_step` 的状态分拣语句块 (`if self.state == ...`)
  - **变更 (重写转移逻辑)**:
    - **当 `EXPLORE` 模式**: 仅在后台出现新特征引发 $I_{loc}$ 出现极大值时，打断探索切入 `INSPECT` 并移动视心至爆发区中心。
    - **当 `INSPECT` 模式**: 视心开始在爆破域作非随机密集震荡。当聚散出的数据使某个已知物体的 $A^{*}_{sem}(t)$ 越过识别阈值 $\tau_{recognize}$ 时，说明对象被初步唤起，切入 `CONSOLIDATE` 去查验；反之若密集眼跳将局部 $I_{loc}$ 耗干依然一无所知说明对象是首次看到，退回 `EXPLORE` (不再阻塞等待)。
    - **当 `CONSOLIDATE` 模式**: 不再盲目猜步子，用获得最高评分激活的 `SemNode` 制出其预测的特征落点，并压实关注。若出现高形变失误造成认知偏差使得该 $A^{*}_{sem}(t)$ 暴降或校验完成造成返航抑制效应（兴趣度跌入谷底），果断跳回 `EXPLORE`。
