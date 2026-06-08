# Memory Graphs 理论框架重构与算法落地规范 (V5.0)

本项目基于人类视觉认知生物学与动力系统交叉视角，针对前序版本的机械循迹和死锁缺陷进行重构。根据最新指示，算法架构被严格解耦为 **`InterestOptimizer` (兴趣场动态优化器)** 与 **`Controller` (状态机与视线控制器)** 两个核心模块。通过 $1+n_1+n_2+n_1$ 个矩阵空间的解耦叠加，实现自然涌现的视觉注意流与循迹能力。

以下为该框架的严谨算法流程、可能存在的问题评价及数学表述。

---

## 零、 框架评估与潜在修正考虑

### 1. 结构优势与评价
- **解耦之美与物理意义明晰**：通过剥离 `InterestOptimizer`，控制器的职能被极大地纯化（只负责根据 $I_{map}$ 落子并建立图拓扑结构）。兴趣图的演变更符合“多物理场叠加”的原理。
- **孤立特征轨迹抑制 (Feature-Specific Inhibition)**：将“轨迹抑制”从单一的“全局扣减”变为对 $n_1$ 和 $n_2$ 个具体特征的“各自抑制矩阵”叠加。这是极具开创性的设计——它意味着系统注视红色区块产生的抑制，**不会**削弱重叠在该位置上方的一条垂直边缘的显著性。这完全符合人类视知觉的**特征特异性适应（Feature-Specific Adaptation）**规律。
- **降级与兼容设计 (Trace Guide 补丁)**：将自发涌现循迹逻辑封装为 $I_{guide}$ 矩阵。如果前置的抑制与期望场不能完美达成平滑追踪，可以在权重组合中临时加强 $I_{guide}$ 矩阵以确保视觉锁边滑移（Smooth Pursuit）平稳执行。

### 2. 潜在问题与解决方案
- **问题：动态矩阵数量极多 ($1+n_1+n_2+n_1$) 带来的计算/显存爆炸**。由于随着记忆库庞大，活跃的 $n_1, n_2$ 数量可能很多个。高分辨率下每个特征都维护一个浮点矩阵会拖垮显存与运算速度。
- **修正规避方案**：通过稀疏张量（Sparse Tensor）或仅在激活门限（Active）以上的节点生命周期内动态分配（Allocate）及其抑制矩阵内存。当神经元衰退至平静时，立即析构其对应的独立抑制场并归零。

---

## 一、 动态兴趣优化器 (InterestOptimizer)

负责多场矩阵叠加出实时的视觉兴趣地貌 $I_{map}$。

### 1. 矩阵构成体系
假定当前处于 Active 态的 $GposI$ （基础特征）对象子集为 $\mathcal{A}_I$ (数量为 $n_1$ )；处于 Active 态的 $GposII$ （高层语义组合）对象子集为 $\mathcal{A}_{II}$ (数量为 $n_2$)。则实时 $I_{map}$ 为下述矩阵的加权求和（权重作为开放超参数）：

$I_{map}(\mathbf{q}) = w_{base} I_{base}(\mathbf{q}) - w_{inh\_I} \sum_{k \in \mathcal{A}_I} I_{inh\_I}^{(k)}(\mathbf{q}) - w_{inh\_II} \sum_{j \in \mathcal{A}_{II}} I_{inh\_II}^{(j)}(\mathbf{q}) + w_{exp} \sum_{k \in \mathcal{A}_I} I_{exp}^{(k)}(\mathbf{q}) + w_{guide} I_{guide}(\mathbf{q})$

*(为保证合规输出，所有累加后可通过 $\text{ReLU}$ 或 $\text{Sigmoid}$ 归一化至 $[0,\infty)$ 区间)*

### 2. 矩阵详细演算规则
**【矩阵一】静态初始场 $I_{base}$ (1个)**
基于 CNN 张量，依照对 $I_{map}$ 的影响机理划分为两大属性子空间：
- 绝对强度特征 $\mathcal{X}_{str}$ (如：边缘幅值、亮度)：$I_{str} = \sum X_{str}$ 
- 连续性特征 $\mathcal{X}_{con}$ (如：色相 Hue、特定纹理变异度)：利用空间均一化/梯度方差演算：$I_{con}(\mathbf{q}) = \sum \exp(-\|\nabla X_{con}(\mathbf{q})\|^2 / 2\sigma_{c}^2)$
$$I_{base}(\mathbf{q}) = w_{str} I_{str} + w_{con} I_{con}$$

**【矩阵二】底层特征的轨迹抑制 $I_{inh\_I}$ ($n_1$ 个)**
当视线落于 $\mathbf{p}_t$ 并激活/识别出属于子集 $\mathcal{A}_I$ 的特征 $k$ 时。它的抑制矩阵受到局部高斯增长，从而表示“我看过这个特征的这一小块了”：
$$I_{inh\_I}^{(k)}(\mathbf{q}, t) = I_{inh\_I}^{(k)}(\mathbf{q}, t-1) + \eta_{inh} \cdot \exp\left(-\frac{\|\mathbf{q} - \mathbf{p}_t\|^2}{2\sigma_{local}^2}\right)$$

**【矩阵三】高层组合对象的抑制 $I_{inh\_II}$ ($n_2$ 个)**
如果视线移动促使 GmemII 中的组合对象 $j$ 处于激活态，该对象整体亦散发“满足感”抑制其宏观势力范围：
$$I_{inh\_II}^{(j)}(\mathbf{q}, t) = I_{inh\_II}^{(j)}(\mathbf{q}, t-1) + \text{GaussianMask}(\mu_{obj}, \Sigma_{obj})$$

**【矩阵四】拓扑空间跨态预期 $I_{exp}^{(k)}$ ($n_1$ 个)**
如果已知语义组合 $j \in \mathcal{A}_{II}$ 已经被唤醒，且依据其内部建构的图谱拓扑已知它在相对坐标 $\Delta \vec{r}_m$ 处存在子特征 $k$。$GposII$ 遂释出“预期”，直接向特征 $k$ 对应的矩阵注入吸引峰：
$$I_{exp}^{(k)}(\mathbf{q}) = \sum_{j \in \mathcal{A}_{II}} \sum_{\text{link}} w_{exp} \cdot \exp\left(-\frac{\|\mathbf{q} - (\mathbf{p}_{anchor}^{(j)} + \Delta \vec{r}_m)\|^2}{2\sigma_{exp}^2}\right)$$
当视窗抵达并确认此预期后，$I_{exp}^{(k)}$ 即刻在该位面清空。

**【矩阵五】动态循迹引导风格矩阵 $I_{guide}$ (1个)**
提取当前注视点 $\mathbf{p}_t$ 被强烈激活的所有特征响应 $M_{resp}$。生成向边缘滑移或远端跳跃的动力差分：
$$M_{periphery} = \text{ReLU}\left( \text{Blur}(M_{resp}) - M_{resp} \right), \quad M_{remote} = \text{ReLU}\left( M_{resp} - G_{\mathbf{p}_t}(M_{resp}) \right)$$
$$I_{guide}(\mathbf{q}, t) = \alpha \cdot M_{periphery} + \beta \cdot M_{remote}$$
*(此项可确保控制器沿着连续边界和色块向外拓长或跳跃。)*

---

## 二、 神经元三态制基础 (Gmem Nodes Engine)

所有的状态与网络链接由 Gmem 记忆池负责演化：
1. **Calm状态 ($A=0$)**：无响应。被注视或预期激活拉升电位，若 $A(t) > T_{excite}$，转为 Active。若 $A(t) > T_{inject}$，反馈给图检索操作。
2. **Active状态 ($A > 0$)**：锁定并维持高电平兴奋数帧，产生强烈的抑制写入与拓扑推算。倒数完成后强制突变坠入 Refractory。
3. **Refractory状态 (不应期, $-1$)**：一段时间内免疫外界任何电位激励与检索匹配。充作生物视疲劳残像，杜绝“看一眼走不开”的死循环。

---

## 三、 状态控制机与眼跳决断 (Controller)

控制器通过调用 Optimizer 解出的 $I_{map}$ 进行位移投机，并根据局部状态切换大网络逻辑（REVIEW 或 LEARN）。

### 1. Saccade: 视点抉择方程
无论处于哪个宏观状态，眼动的发生服从统一跳跃决策：
$$\mathbf{p}_{t+1} = \operatorname{argmax}_{\mathbf{q}} \left[ I_{map}(\mathbf{q}) \cdot \exp\left(-\frac{\|\mathbf{q} - \mathbf{p}_t\|^2}{2\sigma_{\text{jump\_cost}}^2}\right) \right]$$
*(距离惩罚迫使模型进行小幅度寻迹探索。当局部被 $I_{inh}$ 完全压平，$I_{map}$ 极小时，系统会豁免费跳至远端 $M_{remote}$)*

---

### 2. 双周宏观运行态 (Macro-States Flow)

每个新视野优先送入 **REVIEW (复习验证态)**，不能被复习解释的区域残骸（仍然残存高亮 $I_{map}$）下放至 **LEARN (学习态)**。

#### 相位 I: REVIEW 复习验证态
1. **随机预视**: 挑选 $I_{base}$ 的一个显著高点，直接抓取周围特征，触发它们的状态冲入 `Active` $(n_1)$。
2. **广度记忆查询**: 此动作反向刺激了 $GmemII$ 的神经网路进行广度优先搜索。若它们成功拼贴，识别出某个高层对象，导致该对象的 $GmemII$ 神经元跃迁至 `Active` $(n_2)$。
3. **期望发散**: 处于 $A_{II}$ 中的神经元根据自身连接，向 Optimizer 下达预期场 $I_{exp}$ (产生 $n_1$ 个矩阵的变动)。
4. **校验追踪**: 在 $I_{exp}$ 制导下（也就是所谓的循着预填的兴趣山峰走），Controller 开始落子 $\mathbf{p}_{t+1}$。符合预期的特征瞬间被点亮（并扣减 $I_{inh\_I}$），数帧后又跌落 -1 不应期，这引致期望峰值和基底均被清剿。
5. **出口判定**: 预期查验完毕，若图的局部兴趣跌入低谷，判定该区域认识完成。如果某大片区域组合查无此物，跳入 `LEARN` 处理陌生世界。

#### 相位 II: LEARN 新认知块建构态
这是围绕尚未归类、兴趣涌动的高亮残骸发生的循迹式构建。

1. **落定新原点 (Anchor Start)**: $I_{map}$ 剩余最亮点作为新事物的第一瞥，抽取写入 $GmemI$，于 $GmemII$ 建库新对象 `Anchor_Obj`。
2. **自发寻迹与吸积 (Self-driven Tracing Loop)**:
   在此循环中逐帧执行直至破坏条件满足：
   - 依赖 Saccade 方程产生新落点 $\mathbf{p}_{t+1}$。
   - 读取该点的所有底层特征，加入 `Active`，在图 $GmemII$ 的当前 `Anchor_Obj` 下新增相对 `peripheral_links`。
   - 被关注特征 $k$ 向 Optimizer 抛去更新参数， Optimizer 中的 $I_{inh\_I}^{(k)}$ 会被狠狠累加，强行掩埋来时路。而 $I_{guide}$ 会基于 $M_{resp}$ 释放向外拓展晕裙和远方跃迁点的信号。
3. **自动结束与跃断 (Leap & Terminate)**:
   伴随不断的吸收，$M_{local}$ 被 $I_{inh\_I}$ 彻底踩烂，$I_{map}$ 在附近已无利可图。
   由于“不应期”拒绝重复拾取，$I_{map}$ 最大极值只可能通过 $I_{guide}$ 产生的远端相似区提供。一旦 Saccade 点 $\mathbf{p}_{t+1}$ 相对上点发生了空间巨步跃迁（超过滑移允许范围）：
   **打断逻辑执行**：当前物体对象外边缘描摹结束！封存当前 `Anchor_Obj` 及其所有挂载链接，跳跃至下一个 $I_{map}$ 高点的异地，重启 LEARN 的新 Anchor 进行循环。全图 $I_{map} \approx 0$ 时流程全终结。
