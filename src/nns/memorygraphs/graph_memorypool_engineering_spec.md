# Memory Graphs 理论框架重构与算法落地规范 (V6.0)

本项目基于人类视觉认知生物学与动力系统交叉视角，架构被严格解耦为 **`InterestOptimizer` (兴趣场动态优化器)** 与 **`Controller` (状态机与视线控制器)** 两个核心模块。通过引入空间足迹矩阵与神经元状态权重的直接耦合，实现自然涌现的视觉注意流与严谨的循迹能力。

---

## 零、 新增核心框架大类：特征通道属性分拣与态势足迹

### 1. 静态初始场特征类型兼容引擎 (Feature Compatibility & Static Base)
CNN 传入的多模态特征维度可能为 `[1, C, W, H]` 甚至包含未来新扩展矩阵。我们必须实现对各通道如何影响 $I_{map}$的解耦与配置化：
- **属性配置文件 (Hyperparameter Dict)**：为各个特征通道指定 `STRENGTH` 或 `CONTINUITY` 的影响方式。
- **绝对强度特征 $\mathcal{X}_{str}$**：代表边缘、高频亮斑等绝对刺激。它们对初始场的贡献直接与幅值正相关：
  $I_{str}(\mathbf{q}) = \sum_{c \in \text{str\_channels}} w_c \cdot X_{str, c}(\mathbf{q})$
- **连续性特征 $\mathcal{X}_{con}$**：代表色相(Hue)或平滑纹理。它们的意义在于其“空间同质性”而非自身读数大小，需通过计算梯度/局部方差取得分布：
  $I_{con}(\mathbf{q}) = \sum_{c \in \text{con\_channels}} w_c \cdot \exp\left(-\frac{\|\nabla X_{con, c}(\mathbf{q})\|^2}{2\sigma_c^2}\right)$
- **多通道融合底图**:
  $I_{base}(\mathbf{q}) = I_{str}(\mathbf{q}) + I_{con}(\mathbf{q})$

### 2. 生物态介导的特征空间足迹 (Spatial Footprint 代替单纯抑制)
放弃单相扣除逻辑。$Gmem$ 节点的内部激活值 $A_k(t)$ 同时承担向 $I_{map}$ 输入“兴趣渴望”与“视疲劳抑制”的双向表达：
- **普通激活期 ($A_k(t) > 0$)**：模型主动寻求该特征出现的其它域，引导产生正向期望。
- **不应闭锁期 ($A_k(t) = -1$)**：生物视神经陷入疲劳状态。强烈的 $-1$ 权重赋予该特征足迹以极强的排斥惩罚！
$I_{spatial\_I}(\mathbf{q}) = \sum_{k \in \mathcal{A}_I} A_k(t) \cdot M^{(k)}_{resp}(\mathbf{q}) \cdot \operatorname{Mask}_{local}(\mathbf{p}_t)$

### 3. 不应期延迟驻留与缓存封存 (Refractory Retention)
当神经元节点触发阈值进入不应期 ($A_k = -1$)：
- **防疲劳缓存**：它**不能**被移出 $GposI$ 活跃查询池。它必须滞留在其中依靠其 $-1$ 的状态继续为兴趣图制造“抑制区 (厌恶区)”，直到计时器流逝使其回归为平静态 (0)，彻底失去对空间地图的话语权方可被移出。
- **GposII 上层封存**：进入不应期的底层特征节点所属的高级 GmemII 对象暂时封存修改权限，直至底层绝大部分特征重新活跃或更换了新 Anchor。

### 4. 基于眼跳轨迹同质性的连续挂载断言 (Trajectory Homogeneity)
对于 GmemII 新节点的判定废除简单的点间距阈值，使用连线跨越判定：
- 取连续两次眼跳落点 $\mathbf{p}_t$ 到 $\mathbf{p}_{t+1}$。
- 在这两点连线构成的线段（或近缘轨迹）上抽取过渡特征。
- 如果轨迹特征与当前物体的锚点特征相似度并未发生深渊级跌落（未跨越对象裂缝），则判定由于物理表面的平滑推延，应该继续挂载为现存 `Anchor_Current` 的 `peripheral_links`。
- 如果轨迹在某处发生剧烈特征不匹配或响应“悬崖”，意味着跳越到了另一个物理对象，切断当前挂载并新开 GmemII 结构。

---

## 一、 动态兴趣优化器 (InterestOptimizer)

负责解算实时的视觉兴趣地貌 $I_{map}$。

实时 $I_{map}$ 组成为基底加上底/高层足迹及引导的超参加权求和：
$I_{map}(\mathbf{q}) = w_{base} I_{base}(\mathbf{q}) + w_{sp\_I} \sum_{k \in \mathcal{A}_I} I_{spatial\_I}^{(k)}(\mathbf{q}) + w_{sp\_II} \sum_{j \in \mathcal{A}_{II}} I_{spatial\_II}^{(j)}(\mathbf{q}) + w_{exp} \sum_{k \in \mathcal{A}_I} I_{exp}^{(k)}(\mathbf{q}) + w_{guide} I_{guide}(\mathbf{q})$

**矩阵演算细节：**
- $I_{base}$：前述通道分离的加和基底。
- $I_{spatial\_I}^{(k)}, I_{spatial\_II}^{(j)}$：带有神经元激活与不应正负权重特性的足迹场。
- $I_{exp}^{(k)}$：高层下发给底层缺失部分的局部预期吸引峰。
- $I_{guide}$：循迹风格的自动补全矩阵 $I_{guide} = \alpha M_{periphery} + \beta M_{remote}$。
*(利用张量的生命周期作内存管理，只为 $A_k(t) \neq 0$ 的非平静节点分配局部高斯脚印内存)。*

---

## 二、 三态运行神经元网组 (Gmem Nodes Engine)

所有的状态与网络链接由 Gmem 记忆池负责演化：
1. **Calm 状态 ($A=0$)**：无响应。被注视或预期激活拉升电位，若 $A(t) > T_{excite}$，转为 Active。若 $A(t) > T_{inject}$，反馈给图检索。
2. **Active 状态 ($A > 0$)**：锁定并兴奋，积极拉高该特征周边区域图寻找并扩充自身。倒数完成后跌入 Refractory。
3. **Refractory 状态 (不应期, $-1$)**：一段时间内免疫外界激励，但在 Gpos 检索场中释放 $-1$ 负压。彻底屏蔽“一直看同一特征”。完毕后返回 Calm 状态。

---

## 三、 状态控制机与眼跳决断 (Controller)

控制器下发 Saccade 指定与维护图对象切分。

### 1. Saccade: 视点抉择方程
$\mathbf{p}_{t+1} = \operatorname{argmax}_{\mathbf{q}} \left[ I_{map}(\mathbf{q}) \cdot \exp\left(-\frac{\|\mathbf{q} - \mathbf{p}_t\|^2}{2\sigma_{\text{jump\_cost}}^2}\right) \right]$

### 2. 双周宏观运行态 (Macro-States Flow)

#### Phase I: REVIEW 复习验证态
1. 全图选点打乱激发 $GmemI$ 态。
2. 在图神经网络发酵并反馈连通已知语义态 $GposII$。
3. 下放预期引导眼跳 $I_{exp}$，按预设山峰踏过校验。相符即进入 $-1$ 闭关清图，图平息代表审核全部通过。无连通则退给 LEARN。

#### Phase II: LEARN 新认知块建构态
围绕剩余或异样区域进行的连续描摹：
1. **落定新原点 (Anchor Start)**: 从残骸高点起建立 $GmemI$ 并根立新 $GmemII$ Anchor。
2. **自感滑移 (Self-driven Tracing Loop)**:
   - 步进解算新视点 $\mathbf{p}_{t+1}$，压入提取。
   - 调用**轨迹同质性判定算法**。如果在 $\mathbf{p}_t$ 至 $\mathbf{p}_{t+1}$ 之间的所有过渡域特征表现出连续性：追加至 `peripheral_links`。
   - 特征节点的自身状态机让其在 Active 发烧后相继掉入 $-1$，迫使 Optimizer 生成向外的推斥力。眼跳自然沿着未遍历的同质边缘一直爬行。
3. **跃阶断层与自动终了 (Leap & Halt)**:
   - 一旦周边跌入低谷逼出大跃迁，或者连线轨迹发生同质性破裂（越跨异物边界）。
   - 切断 Anchor。针对落脚的彼岸另起新炉灶。全图肃静时任务退出。


# graph_memorypool.py 视知觉重构技术规格书 (V6.0)

## 0. 概述 (Overview)
针对死锁和边界判定单调的问题，架构现已被解耦为 **`InterestOptimizer`** 与 **`Controller`**。
特别加入了对于输入通道属性分类的兼容式 `_base_interest` 生成算法，以及将原先粗暴的兴趣抑制转换为由生物三态神经元介导的 **空间足迹场 (Spatial Footprint)**。同时，为了更精细的形变判断，加入 **基于轨迹同质性** 的拓扑断链判断体系。

保留 interest_map 初始化时压低外围兴趣的部分（该部分置于 Optimizer 的基底场生成中）：
```python
h1, h2 = max(0, H//2-(H_orig-r)//2), min(H, H//2+(H_orig-r)//2)
w1, w2 = max(0, W//2-(W_orig-r)//2), min(W, W//2+(W_orig-r)//2)
self.base_interest[:, :, :w1, :] = 0; self.base_interest[:, :, w2:, :] = 0
self.base_interest[:, :, :, :h1] = 0; self.base_interest[:, :, :, h2:] = 0
```

---

## 1. 单独要点区：底层输入特征通道分类与初始基场构成
为接纳形如 `[1, C, W, H]` 甚至未来的所有自定义通道张量集合，初始化与静态兴趣注入需执行属性映射：

- **[修改点 1.0: 特征属性全局配置注入]**
  - 在初始化处明确超参数大类划分：定义某通道属于 `STRENGTH` 还是属于 `CONTINUITY`。目前如边缘属于 STRENGTH，Hue/纹理属于 CONTINUITY。
- **[修改点 1.1: `initialize_base_interest` 算法扩写]**
  - **基于 STRENGTH 的通道**:
    $$I_{str} = \sum_{c \in \text{strength}} w_{c} X_{c}$$
  - **基于 CONTINUITY 的通道**:
    以类似 Sobel 或直接的差分手段求取分布梯度梯级 $\|\nabla_c X_{c}\|^2$：
    $$I_{con} = \sum_{c \in \text{continuity}} w_{c} \exp(-\|\nabla_c X_c\|^2 / 2\sigma_c^2)$$
  - 基场合成为 $I_{base} = I_{str} + I_{con}$ 压入引擎作为地形地貌本底。

---

## 2. 三态制神经元与不应期滞留引擎 (Neuron Dynamics)

- **[修改点 2.1: 追加神经元动态状态与不应期滞留/缓存]**
  - 节点新增 `activation_level` $A(t)$、`state_flag` 和 `timer` 倒数。
  - **状态演变**：
    - `ACTIVE` 态 ($A>0$): 提供正向激活值参与制导场。计时若结束则将其态跌入 `REFRACTORY`，值为 $-1$，同时重置另一个长倒数。
    - `REFRACTORY` 态 ($A=-1$): 执行 `timer -= 1`，倒数完毕状态恢复并切为 `CALM` ($A=0$)。
  - **[核心机制 - 滞留场中投递抑制]**: 当神经元处在 `-1` 态，**坚决不从 Gpos 查询活跃检索池内剔除它**。由于它的激活值为强负值，它在 Optimizer 中构筑起强大的排斥足迹！在上层 $GmemII$，对掉入不应期的旧特征下属语义节点实行挂载权限封存，直到底层特征大部分重构或更换了新 Anchor。
- **[修改点 2.2: Gpos注入检索屏蔽门限]**
  - 不应期节点仍能提供 $I_{map}$ 投射。但其在**被当前视窗提取查询匹配时**，将被筛除拒绝对上游匹配联结（即抽取满足 $A(t) > T_{inject}$ 且 state_flag != REFRACTORY 的节点以进入图检索）以防视察幻觉。

---

## 3. InterestOptimizer 模块的空间足迹场改造

摒弃死板抑制，将 `I_inh_I` 及 `I_inh_II` 升级为结合神经元激活值 $A$ 的空间足迹 $I_{spatial\_I}$ 和 $I_{spatial\_II}$：

- **[修改点 3.1: 耦合节点状态激活权值的加权求和计算]**
  把特征自身的 $A_k(t)$ 提取出来做场的有符号系数：
  $$I_{spatial\_I}^{(k)}(\mathbf{q}) = A_k(t) \cdot M^{(k)}_{resp}(\mathbf{q}) \cdot G_{\text{local}}(\mathbf{p}_t)$$
  对于激活状态的正极 $A_k > 0$（活跃），场正向引流（期望聚集）；而当其倒数进入不应期变为 $-1$，该局部直接塌陷成负值抑制区，实现了用特征本身状态流形天然表达防重叠视疲劳的生物本能。
- **$I_{map}$ 实时呈现**:
  $$I_{map} = w_{base} I_{base} + w_{sp\_I} \sum I_{spatial\_I} + w_{sp\_II} \sum I_{spatial\_II} + w_{exp} \sum I_{exp} + w_{guide} I_{guide}$$

---

## 4. Controller 基于眼跳轨迹同质性的断链挂载法则

摈弃由距离过长判断边界（这种判据极为脆弱）。

- **[修改点 4.1: 对象挂载切断的动态连线判定]**
  - 在 `LEARN` 阶段两点跃迁 $\mathbf{p}_t \to \mathbf{p}_{t+1}$ 时，验证这段眼跳轨迹：
  - **判定执行**: 提取 $\mathbf{p}_t$ 到 $\mathbf{p}_{t+1}$ 轨迹连线中途的特征样本。计算这些特征同 $GmemII$ 中当前 `Anchor_Current` 的相似性（即便使用低复杂度运算如判断是否出现相似度断层深渊）。
    - **同质挂载**: 如果未出现跨越性深落，即轨迹上特征跟物体的关联仍然被保持，视同此动作没有跨越物体边界，仍然在表面游走追踪。顺势接续挂载至 `peripheral_links` 内，GmemII 继续攀加延发。
    - **异质切断**: 若轨迹在某处发生了特征悬崖/严重不相似（说明视界越过了物体边界，落到了背景环境上再返回部分边缘），跨越了真实物理边界。此时不论空间远近强制**封死当前 `Anchor`** 切断联系，就地单立一套新 $GmemII$ 纪元。
