# Memory Graphs 算法重建方案 (v2.0)

针对原算法中状态机僵化及视觉注视停滞的问题，基于技术团队及您的最新反馈，新算法引入生物神经网络的“三态神经元及不应期”机制与大幅简化的“复习-学习”双相模式。以下是对新算法的客观评价、可能存在的问题及完善方案，最后给出严谨的算法全流程与数学表述。

## 零、 新算法评价与完善建议

### 1. 优势与合理性评估
- **生物学合理性强**：引入“激发 $\to$ 不应期 $\to$ 平静”的三态机制，契合生物视觉系统的抑制与防抖机制，这从根本上杜绝了此前模型对单一兴趣点无限循环死锁的缺陷。
- **状态机极简、鲁棒**：将脆断且彼此纠缠的“探索-审查-巩固”链条剥除，代之以“先尝试匹配关联（复习），失败则增量建库（学习）”的逻辑，这是典型的贝叶斯/预测编码认知模式。
- **自适应的空间注意力引擎**：借助单连通域执行“局部抑制（已看）”与“孤岛凸显（相似但未看）”，使得眼跳轨迹（Saccade）从被动的“瞎跑”变成了具备主动收割完整物体特征意图的连续流。

### 2. 潜在问题与细节补全
- **关于 GmemII 连续写入的拓扑结构 (LEARN)**：
  - **问题**：如果学习阶段“每写入一个特征都直接互相链接更新 peripheral_links”，可能会由于连续眼动的顺序性引发线性的“一字长蛇阵（A-B-C-D）”，而非对单一物体的星型描述。
  - **补全**：为该次 LEARN 阶段确立一个**语义锚点 (Anchor)**（即第一眼的质心）。在兴趣耗尽前的这个连续学习周期内，所有提取出的周围特征均计算相对于它的空间极坐标变换，统一挂载在 Anchor 的周边结构上。
- **图广度优先搜索的条件与爆炸问题 (REVIEW)**：
  - **问题**：随机抽取几个孤立特征（$n_1, n_2, \dots$）来逆向宽搜它们在 GmemII 上的相关记忆，如果图巨大可能会引发组合爆炸。
  - **补全**：在 $G^{II}$ 的拓扑上限定搜索深度 $k$。如果这些被抽样的特征点对能在 $k$ 跳最短路径 (Shortest Path $D \le k$) 内收敛汇聚到同一个已知 Semantic Node 上，则认定为该对象；否则直接剪枝，判定为“不认识的新组合”，触发复习失败。
- **单连通区域的在线分割成本**：
  - **问题**：在每一次眼跳去提取像素级的单连通掩膜对于实时图计算开销很大。
  - **补全**：通过空间衰减高斯掩膜叠加特征激活的 Gpos 响应图，进行快速近似，截断出周边属于同一事物的连续高响应区。

---

## 算法全流程与严谨数学表述

### 一、 神经元三态与激活阈值模型 (Neuron Dynamics)

每一个存储在 $G_{mem}^I$ 及 $G_{mem}^{II}$ 的特征节点 $n$ 拥有一个离散-连续混合状态标志 $S_n(t)$ 及内部计时器 $c_n(t)$。

1. **状态定义**：
   $$S_n(t) \in \{ 0 \text{ (Calm)}, \text{val}>0 \text{ (Active)}, -1 \text{ (Refractory/Closed)} \}$$
2. **刺激与演化动力学**：
   设 CNN 或前层图的局部输入对其产生了刺激强度 $E(t)$。
   
   - **闭合态 / 不应期演化**：当 $S_n(t-1) = -1$ 或者节点处于强制兴奋期（即计时器 $c_n > 0$），此时节点**免疫任何外界激活 $E(t)$**。倒计时器随帧衰减 $c_n(t) = c_n(t-1) - 1$。
      - 若兴奋期结束，状态跌落至不应期：$S_n \gets -1$，并重置计时器 $c_n \gets \tau_{\text{refractory\_dur}}$。
      - 若不应期结束：恢复平静 $S_n \gets 0, c_n \gets 0$。
   - **静息态 / 普通激活演化**：当 $S_n(t-1) \ge 0$ 且无强制限制时：
     $$A_n(t) = A_n(t-1) \cdot \lambda_{\text{decay}} + E(t)$$
   - **过载门限 ($T_{\text{excite}}$)**：
     若刺激累加超过兴奋门限 $A_n(t) \ge T_{\text{excite}}$，神经元被彻底激活：
     状态锁定为强兴奋（保持一段时间），期间激活值维持最大，免疫外界影响：
     $c_n \gets \tau_{\text{active\_dur}}, A_n(t) \gets A_{\text{max}}$。对于未达 $T_{\text{excite}}$ 的，其激活量则随时间自然衰退。

3. **Gpos 检索门限 ($T_{\text{inject}}$)**：
   只有当激活值超越检索界限且未处于不应期的节点，才会向 Gpos 层注入光栅去映射自身坐标：
   $$G_{\text{inject}}(t) = \left\{ n \ \mid A_n(t) \ge T_{\text{inject}} \ \land \ S_n(t) \neq -1 \right\}$$

---

### 二、 动态兴趣图驱动引擎 (Dynamic Interest Map)

系统维护全局浮点矩阵 $I_{\text{map}} \in \mathbb{R}^{H \times W}$，启动时 $I_{\text{map}} = I_{\text{base}}$。不再单纯依赖硬编码高斯，而是基于 Gpos 空间动态刻画。

1. **抑制与远端拉高 (Inhibition of Return & Distal Excitation)**：
   LEARN 或 REVIEW 阶段，若视点中心落在 $\mathbf{p} = (x,y)$ 且激活了特征 $n \in G^I$，该特征注入 Gpos 得到了全图响应矩阵 $M_{\text{sim}}$。
   以 $\mathbf{p}$ 为原点，根据空间距离连通关系对 $M_{\text{sim}}$ 采取二值/平滑分割，分为：
   - 包含注视点的单连通域掩码：$M_{\text{local}}(\mathbf{q}) \in [0,1]$
   - 视点以外的远端孤立高响应掩码：$M_{\text{remote}}(\mathbf{q}) \in [0,1]$
   
   实时更新公式：
   $$I_{\text{map}}(t) = \max\left(0,\ I_{\text{map}}(t-1) - \alpha \cdot M_{\text{local}} + \beta \cdot M_{\text{remote}}\right)$$
   使得注视局部被深挖耗尽后，强迫注意力迁跃向尚未观察的相似目标碎片。

2. **视点抉择 (Saccade Target Selection)**：
   新视点的选择综合兴趣剩余量与眼跳代价（距离跳跃惩罚）：
   $$\mathbf{p}_{\text{new}} = \arg\max_{\mathbf{q}} \left[ I_{\text{map}}(\mathbf{q}) \cdot \exp\left(-\frac{\|\mathbf{q} - \mathbf{p}\|^2}{2\sigma_{\text{saccade}}^2}\right) \right]$$

---

### 三、 算法流：复习与学习周期 (State Machine: REVIEW & LEARN)

模型启动处理单张图像时，初始过程必须是 **REVIEW $\to$ LEARN**。

#### 阶段 1：复习主导 (REVIEW STATE)
尝试利用既有记忆图解构当前画面，预取期待。

1. **粗提取**：随机（或依最大的 $I_{\text{base}}$）挑选一个起始注视点。在注视域周边抽样 $m$ 个特征点，灌入 $G^I$ 形成局部点集 $\mathcal{X}_{local}$，由于刺激大直接导致相关神经元激活。
2. **广度优先图校验 (BFS GmemII Query)**：
   - 在图 $G^{II}$ 拓扑搜寻：对 $\mathcal{X}_{local}$ 中任意配对点，搜索在限定步数 $k$ 内能否共同寻访到某以已知语义中心节点 $Sem \in G^{II}$。
   - **CASE A：无法寻址连通 (Unrecognized)**
     如果特征组合不在任何 $k$-跳邻域内，判定这一堆特征无法靠当前记忆重组解释（不认识），立即终止对该区域的 REVIEW，释放给 LEARN 处理。
   - **CASE B：唤醒记忆 (Recognized)**
     确定了中心节点 $Sem$。据其在 $G^{II}$ 的内存（如其它附属特征期望的 $(\rho_i, \theta_i)$），向 $I_{\text{map}}$ 的对应极坐标投影位置强行注入期望高峰（拉高兴趣），引导系统验证。
3. **退回与状态结束**：验证成功的特征将被压入**不应期 ($S_n = -1$)**，从而由于免疫再次提取使得它所在的 $I_{\text{map}}$ 快速平息。当全图 $I_{\text{map}}$ 被消耗清空（全认识），处理完毕；若部分较高但无法被认知，切入 LEARN 状态。

#### 阶段 2：学习主导 (LEARN STATE)
处理 REVIEW 阶段撇下的高兴趣残留，进行物块边缘拓扑刻写。

1. **锚点构建 (Initial Anchor Placement)**：
   接管高兴趣注视点 $\mathbf{p}_0$，将其强行激活。把 $\mathbf{p}_0$ 对应特征写入 $G^I$ 及建立全新 $G^{II}$ Semantic根节点 $N_{\text{anchor}}$。
2. **连贯滑移吸收 (Topological Sweep Loop)**：
   - 当前注视点 $\mathbf{p}_i \to$ 提取视点内新特征并写入 $G^I$ $\to$ 判断 $T_{\text{inject}}$ 并置入 Gpos。
   - 提取它相对 $N_{\text{anchor}}$ 的变换（极坐标距离等），直接将边缘连接挂载进 $N_{\text{anchor}}$ 的 `peripheral_links`。
   - 使用数学部分 [二、1] 的规则，抑制当前单连通部分 $M_{\text{local}}$，拔高 $M_{\text{remote}}$。
   - 依照数学部分 [二、2] 的带有位移惩罚的机制生成新一步的注视点 $\mathbf{p}_{i+1}$ （因为存在距离衰减，系统会优先把近处的连通表面扫除干净）。
3. **状态衰弱与脱退**：
   随着眼跳，该新生对象的物理空间被扫描耗散掉兴趣，大量特征依次陷入 $-1$ 闭合不应期。当 $I_{\text{map}}$ 该区域全部榨干，这组关于该新物体的知识彻底进入 Gmem 二级缓存拓扑。LEARN 结束，若有其他残存兴趣区，转回复习。

# Memory Graphs 算法重建方案 (v2.1)

基于现有的神经元三态机制、学习/复习双相状态机基石，针对“连续学习时的边界确定”以及“Gpos激活区域的空间划分（不再要求严格单连通）”的探讨，对算法细节进行深化与修正：

## 核心重构与问题讨论

### 一、 学习阶段的边界问题 (Object Boundary Determination)

**问题描述**：在 LEARN 阶段，将第一眼注视点确立为 Anchor 后，如果视点不断按照 interest_map 移动并把新特征挂载到该 Anchor 上，系统如何知道“这个物体已经看完了”，从而停止挂载并可能为下一个物体建立新的 Anchor？

**解决方案：基于“渐进滑移”与“空间跃迁”的界定**

边界的判定不需要依赖预先的语义分割，而是通过**注视点轨线的连续性**与**局部特征激活谷底**自然涌现的：

1. **兴趣环形扩张 (Peripheral Sliding)**：
   由于新算法中会“压低当前激活区，拉高其外围”，这天然引导了下一步眼跳通常落在当前特征的“边缘/相邻区域”。只要下一步眼跳 $\mathbf{p}_{i+1}$ 落在这个被拉高的“近邻外围（Periphery）”内，我们就认为视线沿着同一个物体的表面或边界在**滑移**，此时提取的新特征继续挂载到当前的 Anchor 上。
2. **能量谷越界与跳跃 (Spatial Leap & Valley Crossing)**：
   假设某次沿外围的眼跳，提取的新特征 $\mathbf{p}_{i+1}$ 注视时，其所在的局部激活区域与原 Anchor 及其已知的 Peripheral 集合之间，在底图中出现了一道“低响应沟壑”（即中间跨越了不属于该物体特性的抑制区）；或者，当前的局部兴趣被完全耗尽，视点**跃迁 (Leap)** 到了被拉高的“远端激活区域（Remote）”。
3. **边界判定条件**：
   引入连续挂载的**打断机制**：当新的注视点 $\mathbf{p}_{i+1}$ 距离上一个点 $\mathbf{p}_i$ 超过某个由于连续结构延伸所允许的阈值（或者 $\mathbf{p}_{i+1}$ 落在了当前 Anchor 所有已有特征点构成的空间高斯包络之外），系统判定为**越界**。
   此时：**封闭当前 Anchor**，如果在 LEARN 状态下兴趣图依然有高点，则将 $\mathbf{p}_{i+1}$ 作为一个**全新的 Anchor**，另起一个 GmemII 拓扑对象。

### 二、 Gpos注视区域的空间划分法 (Spatial Proximal vs. Distal Regions)

**原设定修正**：不需要严格执行图形学上的“单连通域”查找（开销大且易受噪声干扰）。我们只需利用带有距离衰减的空间包络，将 Gpos 响应矩阵 $M_{\text{resp}}$ 软分割为“注视点所在的局部”、“局部的外围扩展带”以及“注视点不在的远端区域”。

**数学定义（软分割计算）**：
设当前特征经 Gpos 的全局激活响应矩阵为 $M_{\text{resp}}(\mathbf{q})$，当前注视点为 $\mathbf{p}$。

1. **局部注视激活区 (Local Activated Region)**：
   利用以 $\mathbf{p}$ 为中心的高斯掩膜截取局部响应：
   $$M_{\text{local}}(\mathbf{q}) = M_{\text{resp}}(\mathbf{q}) \cdot \exp\left(-\frac{\|\mathbf{q} - \mathbf{p}\|^2}{2\sigma_{\text{local}}^2}\right)$$
2. **局部外围区 (Peripheral Halo Region)**：
   对其进行空间扩散 $\text{Blur}(M_{\text{local}})$（例如使用平滑核），减去原局部区，得到边缘带掩膜：
   $$M_{\text{periphery}}(\mathbf{q}) = \max\left(0, \text{Blur}(M_{\text{local}})(\mathbf{q}) - M_{\text{local}}(\mathbf{q})\right)$$
3. **远端激活区 (Remote Activated Region)**：
   其余的强响应区即为注视点不在的远端：
   $$M_{\text{remote}}(\mathbf{q}) = \max\left(0, M_{\text{resp}}(\mathbf{q}) - M_{\text{local}}(\mathbf{q})\right)$$
4. **Interest Map 的动态重塑**：
   根据最新讨论的要求：压低注视点所在的激活区，拉高其外围以及注视点不在的远端激活区。
   $$I_{\text{map}}(t) = \max\left(0, I_{\text{map}}(t-1) - \alpha \cdot M_{\text{local}} + \beta \cdot M_{\text{periphery}} + \gamma \cdot M_{\text{remote}}\right)$$
   *(通过这种高斯包络扩散的方法，既降低了算力开销，又实现了“沿边缘蔓延搜寻”或“跃迁寻找同类物体”的视觉注意力导向。)*

---

## 完整算法流程总结 (Algorithm Flow V2.1)

### Phase 1: 神经元状态演化层 (随每一帧时间步 t 后台运行)
每个时间步更新所有 $n \in G^I, G^{II}$ 节点：
- **状态变量**：$S_n(t) \in \{0, \text{val}>0, -1\}$，计时器 $c_n(t)$。
- **更新法则**：
  - `If` $S_n(t-1) == -1$ (不应期): 免疫外部刺激 $E(t)=0$。随计时器 $c_n$ 倒数，倒数结束恢复至 $S_n=0$ (平静态)。
  - `Else`: 接收特征刺激，刷新激活值 $A_n(t) = A_n(t-1) \cdot \lambda_{\text{decay}} + E(t)$。
  - `If` $A_n(t) > T_{\text{excite}}$: $S_n \to \text{Active}$。锁定并进入强兴奋计时，计时结束后强制转为 $-1$ (闭合不应期)。
  - `If` $A_n(t) > T_{\text{inject}}$: 达到注入门槛。提取该节点的特征向 Gpos 进行反馈激活，生成 $M_{\text{resp}}$。

### Phase 2: REVIEW 复习状态 (看图时优先进入)
1. **预查验 (Sampling)**：随机选定（或基于图像本身的显著度 $I_{\text{base}}$）一个起始注视点。提取该注视域周边的若干特征集合 $\mathcal{X}_{loc}$。刺激使对应 $G^I$ 节点迅速越过 $T_{\text{excite}}$ 阈值激活。
2. **广度记忆查询 (Graph BFS)**：
   - 以该批激活的 $G^I$ 的特征为起点，在 $G^{II}$ 进行深度限制 $k$ 的最短路径搜索。
   - **若失败（新物体）**：找不到能统一收拢这批特征的共同语义节点 $Sem$ $\to$ 判断为“不认识”，直接终止当前局部的 REVIEW 操作，转储给 LEARN 状态处理。
   - **若成功（已知物体）**：找到共享的已知语义节点 $Sem$。逆向读取 $Sem$ 中记载的其余特征部的“相对期望坐标”。在这些预期位置，向 $I_{\text{map}}$ 直接添加高斯兴趣峰，制导眼跳。
3. **闭眼与确认 (Confirmation & Dimming)**：
   视觉被引导至峰值点，若提取特征验证相符，该特征立刻冲入强激活态后进入 $-1$ 闭合态，并在 $I_{\text{map}}$ 上彻底抹平该处的探究兴趣。
4. **状态转移出口**：
   - 验证结束全图 $I_{\text{map}}$ 回归极低水平 $\to$ “全看懂了”，继续随小概率随机游走或待机。
   - 包含着不可解读的残余大片兴趣高点 $\to$ 无法被已知结构解释，自动切入 LEARN 状态。

### Phase 3: LEARN 学习状态 (拓扑构写)
接管 REVIEW 阶段遗留或放弃的陌生事物兴趣区。

1. **设定锚点 (Set Anchor)**：视线被吸附到全图最高 $I_{\text{map}}$ 处 $\mathbf{p}_0$。提取该点特征写入 $G^I$，无条件触发其高强激活态。同时在 $G^{II}$ 中新建一个独立的根节点 `Anchor_A`。
2. **学习闭环 (Observe-Write-Leap Loop)**:
   在此循环阶段中，眼跳（通常1步为1帧）在未耗尽兴趣前持续运行：
   - **感知写入**：当前视点 $\mathbf{p}_i$，提取特征存入 / 激活 $G^I$ 节点，并触发 $Gpos$ 生成 $M_{\text{resp}}$。
   - **掩膜重塑**：依据 $M_{\text{resp}}$ 和 $\mathbf{p}_i$ 算得上文公式中的 local/periphery/remote 三分掩膜。依据公式更新 $I_{\text{map}}(t)$，强力压制当前点，推叠周边和远端。
   - **挂载判定 (Boundary Check)**：
     - 使用新的 $I_{\text{map}} \times \exp(\text{Distance\_Penalty})$ 确定下一步要看的点 $\mathbf{p}_{i+1}$。
     - 若 $\mathbf{p}_{i+1}$ 距 $\mathbf{p}_i$ 属于滑移（即落入近距离的外围推高区），则将 $\mathbf{p}_{i+1}$ 的特征转换为相对极坐标，更新至 `Anchor_A.peripheral_links`，图块正常生长。
     - 若 $\mathbf{p}_{i+1}$ 发生空间跃迁（跳至遥远 Remote，跨度 $> D_{thres}$），则**闭合 Anchor_A** 不再挂接后续特征。此时若学习模式未结束，系统会将 $\mathbf{p}_{i+1}$ 作为新对象，为其创建 **`Anchor_B`**。
3. **退潮 (Cool Down)**：
   扫描过的表面随着神经元相继进入 $-1$ 不应期而在底层平息。当所有陌生区域都被扫清，$I_{\text{map}}$ 趋于空白，学习阶段完成。系统可主动退回 REVIEW 状态（或等待下一张图像刺激）。


# graph_memorypool.py 新算法重构技术规格书 (V2.1)

## 0. 概述 (Overview)
针对先前的机械状态死锁以及过度依赖状态机制划定对象边界的问题，将原有的（EXPLORE/INSPECT/CONSOLIDATE）替换为极简且具备生物学特性的双相模式（REVIEW/LEARN）。并引入具有“平静-激活-不应期”的三态神经元机制，配合基于空间距离衰减的高斯软划分自适应兴趣图（Interest Map），实现连续平滑的注意力滑移与对象特征挂载断开自发判定边界。

技术团队需要在修改时完全摒弃旧版本的 SaccadeState 及其判定流，严格遵循本新规格书进行代码结构的颠覆。

---

## 1. 神经元三态机制与生命周期控制 (Neuron Dynamics)
在 `GmemI` 和 `GmemII` 的存储神经元中剥离静态数据堆积的概念，融入带有生物激活特征、锁定阈值、和“不应闭合期”的混合状态引擎。

- **[修改点 1.1: 追加神经元动态状态位]**
  - **位置**: `ModalityNode` (`GmemoryI`) 与 `SemanticNode` (`GmemoryII`) 类声明。
  - **新增结构**:
    - `self.activation_level`: 浮点数，代表神经元活跃电位（0 代表平静，>0 代表激活强度，-1 被预留或者通过额外 flag 标记此神经元处于不应期）。
    - `self.state_flag`: 枚举/整形标记态，例如 `CALM (0)`, `ACTIVE (1)`, `REFRACTORY (-1)`。
    - `self.timer`: 整数倒数器，用于在进入极度兴奋态及不应期时的时域锁定（通常随步进帧每次减1）。
  
- **[修改点 1.2: 时域状态演化核（时间流速引擎）]**
  - **位置**: 在 `GmemoryI` 和 `GmemoryII` 模块层抛出一个全局时间推演方法 `tick_update(E_input_dict)`。
  - **变更 (状态机推进)**:
    - `If state_flag == REFRACTORY`: 完全屏蔽对该节点的外部刺激，仅仅 `timer -= 1`，如果 `timer <= 0` 状态切为 `CALM`。
    - `If state_flag == ACTIVE` 且 `timer > 0`（处在强激发强锁定期）: 忽略输入刺激，仅仅 `timer -= 1`，如果倒数结束，强制将其状态跌入闭合防抖屏蔽态 `state_flag = REFRACTORY`，重置为另一个 `timer` 倒数。
    - `If state_flag == CALM` 或 无锁定期普通兴奋：接受网络层该帧算出的此神经元的激活输入 $E(t)$ 增量，$A(t) = A(t-1) \times \lambda_{decay} + E(t)$。一旦 $A(t) > T_{excite}$，神经元爆走，变为 `ACTIVE`，激活定死为最大值并锁入倒数 `timer = \tau_{active}`。

- **[修改点 1.3: Gpos注入检索的屏蔽门限]**
  - **位置**: 对接 `Gpos` 的 L1 检索池抽取口。
  - **变更**: 在执行抽取用于激活并生成 $M_{resp}$ (即生成返回结果并用于路由投影的热图) 时，必须筛去不达标和闭锁节点：必须满足 $A(t) > T_{inject}$ **且** `state_flag != REFRACTORY` 的节点，防抖和背景杂噪借此被干净过滤。

---

## 2. 状态机精简重构：学习与复习引擎 (REVIEW & LEARN Phase)
状态切分由原本的主观推测彻底转为先检验匹配度，失败则建图学习。

- **[修改点 2.1: Controller 主状态精简]**
  - **位置**: `Controller` 类的 `__init__` 与 `run_step` 主干框架。
  - **变更**: 删除 `SaccadeState` 中旧态，将模型置于 `REVIEW` 或者 `LEARN` 主分支运行。模型启动单幅图片通常设定为以 `REVIEW` 启动。
  
- **[修改点 2.2: REVIEW 阶段逻辑重写]**
  - **过程逻辑**:
    1. **抽样激活**: 每执行帧系统随机或沿着基准 $I_{base}$ 制导找一次落点注视，从中硬提取并推送几个特征直接达到 $GmemI$ 中的强激活门限。
    2. **广度验证查找 (图上 BFS)**: 对着这批特征（对应的 $GmemI$ Nodes），在 $GmemII$ 的现有连结拓扑上执行限制最大深度为 $k$ 跳的宽搜。若某一条或多条路径能回溯导向特定的某个已知 `SemanticNode`，认定为【认识事物】。
    3. **制导校验推送**: 读取该认识的 `SemanticNode` 麾下的其他特征节点的“预期极坐标偏移”，直接反映并在全局 $I_{map}$ 上这些预期坐标点拉高高斯峰（注入“期盼值”引流注意），一旦后续眼跳视点抵达并证实，对应特征飙升后即刻步入闭锁态 `REFRACTORY`。
    4. **下放与换挡**: 若特征集合在图上根本无路可通（BFS 无法收敛至单点或无路径），当前区域确认为“陌生”，释放并即刻将环境运行态转储至 `LEARN` 处理。

---

## 3. 动态兴趣掩膜与视线引擎 (Dynamic Interest Map Driver)
新算法摒弃了旧有的只是硬叠加的方式。需要利用当前被看到的特征响应作距离扩散软分割，拉高远处压扁已看中心。

- **[修改点 3.1: interest_map 驱动算法彻底重构]**
  - **位置**: `Controller` 的 `post_step_update_interest(cx, cy)` 和后续生成新注视点的模块。
  - **变更 (引入注视中心距掩膜分离)**:
    借由 Gpos 返回给当帧的全局热图 $M_{resp}$ 和当前视点坐标 $\mathbf{p} = (cx, cy)$ 进行三种图划分（利用带参高斯衰减算子运算）：
    1. 计算局部掩膜: $M_{local} = M_{resp} \times G_{\sigma_{local}}(\mathbf{p})$
    2. 计算外围晕带: $M_{periphery} = \text{ReLU}(\text{Blur}(M_{local}) - M_{local})$ （通过简单的高斯滤波扩张原区域再减去自身即得出边缘圈）
    3. 计算远端响应: $M_{remote} = \text{ReLU}(M_{resp} - M_{local})$
  - **增损兴趣图方程 (核心修改)**:
    覆写下一帧用的完整地图状态：
    $$I_{map}(t) = \text{ReLU}(I_{map}(t-1) - \alpha \cdot M_{local} + \beta \cdot M_{periphery} + \gamma \cdot M_{remote})$$

- **[修改点 3.2: 基于距惩罚的制导跳转函数]**
  - 眼跳目标位置函数修改为：$\mathbf{p}_{new} = \text{argmax}_{\mathbf{q}} [I_{map}(\mathbf{q}) \times \exp(-\|\mathbf{q} - \mathbf{p}\|^2 / 2\sigma^2_{saccade})]$，权衡兴趣浓度与步径惩罚。

---

## 4. LEARN 阶段的物体边界与平滑拓扑挂载 (Topological Trace-Mounting)
将拓扑的生成绑定到“视角的滑移过程（连续连通性）”上，并由突增跳跃打断充当换物边界，废除原死循环的 `CONSOLIDATE`。

- **[修改点 4.1: 对象滑移锚定写循环]**
  - **变更**: 在 `LEARN` 主支路的入口：
    当接收到一个残破高热兴趣点 $\mathbf{p}_0$ 后，提取其 $GmemI$ 写入，并在 $GmemII$ 初始化一个 `Anchor`_Current。
  - 随后框架在 `LEARN` 阶段以 **Observe-Write-Leap** 回路的逐帧持续跟进：
    1. 根据 [3.2修改] 移指的 $\mathbf{p}_{new}$，写入对应点的 $GmemI$。
    2. **连贯检定**: 比较两次落点距离 $\Delta D = \|\mathbf{p}_{new} - \mathbf{p}_{old}\|$。如果 $\Delta D < D_{thres}$，判定视线仍借由被推高的局部裙带外围 ($M_{periphery}$) 在同一事物表面**滑坡扫略**。
    3. **追加拓扑**: 视之共属于同一对象体，计算此时 $\mathbf{p}_{new}$ 相对建立的 `Anchor`_Current 的空间相关映射系，直接生成极坐标/正交补偿作为相对位写入到 `Anchor`_Current 的 `peripheral_links` 内，连通其子枝。

- **[修改点 4.2: 边界剥离和新物件起锚判定]**
  - **变更**: 在持续挂载过程中，若算出的某一步落点跃迁 $\Delta D > D_{thres}$ （往往是由于眼前的局部兴趣全被剥干打入 `REFRACTORY` 封闭态后，图将视点推逼到了在图内响应远端 $M_{remote}$ 的另一处相似破片）。
  - 这表明旧物体结构已被**封闭勘探完结**。代码当即闭口当前保留完好的 `Anchor`_Current。并将这一远端点 $\mathbf{p}_{new}$ 设立为新的生发对象根源位 `Anchor_New` 重复生发扫描直至画面失去所有高亮兴趣源切回闲置。
