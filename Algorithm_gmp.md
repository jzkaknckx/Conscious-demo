# 视觉认知与记忆流形重构算法规范 (Algorithm_gmp)

本模块聚焦于 `graph_memorypool.py` 中最基础、原生的数据结构——记忆图谱池（Gmem）与空间动力投影网（Gpos），并详述两者交互的精确数学定义。这是更高阶场驱动与宏微观眼跳引擎运转的地基。

## 一、 分层分布记忆网池 (Gmem: Modality & Semantic Nodes)

Gmem 采用物理脱钩的双层拓扑设计。底层 `ModalityNode` (GmemI) 直指感知孤立块特征；上层 `SemanticNode` (GmemII) 不保存像素数据本身，仅存取指向对象的指针与极坐标骨架结构（外围拓扑约束）。一切神经元节点的流形生命周期均服从统一的状态机。

### 1. 神经元激活衰减状态机 (Neuron Dynamics)
每一个在池内生根的节点记录当前的激活电位 $A_t$ (`activation_level`)、枚举标志态 $S_t$ (`state_flag`) 与疲劳计时器 $\tau_t$ (`timer`)。
其状态变迁依赖输入的外部能量 $E_{input}$ 并遵循三极稳态模型：

- **平复渴求态 (CALM, $S_t = 0$)**:
  在安静环境缺乏关注度时等待能量输入，具有记忆时延衰减系数 $\gamma$（分 `decay_rate_I` 与 `decay_rate_II`）：
  $$A_{t+1} = \gamma \cdot A_t + E_{input}$$
  若电位受视觉关注而抬升一旦越过起燃阈值（$A_{t+1} > T_{excite}$），节点神经元状态将发生跳跃，切入并锁定为 **ACTIVE**：
  $$S_{t+1} = 1, A_{t+1} = 1.0, \tau_{t+1} = \tau_{active}$$

- **高燃辐射态 (ACTIVE, $S_t = 1$)**:
  神经元电位锁定为满负荷正极 $A_{t+1} = 1.0$，向 Gpos 发出强查询，同时计时器递减：$\tau_{t+1} = \tau_t - 1$。
  不可逾越的生理法则在此生效：当计时器燃尽（$\tau_{t} \le 0$），将不可阻挡地跌入不应期，抛出厌恶负能量。
  $$S_{t+1} = -1, A_{t+1} = -1.0, \tau_{t+1} = \tau_{refractory}$$

- **闭锁冰川态 (REFRACTORY, $S_t = -1$)**:
  隔离任何外界电位 $E_{input}$，电位锁定在坚硬的深负极处 ($A_t = -1.0$) 拒绝重复探索且强制抹除周边场。直至倒数熬出封锁线 $\tau \le 0$：
  $$S_{t+1} = 0, A_{t+1} = 0.0$$

### 2. GmemI 孤立特征节点 (ModalityNode)
GmemI 定义了视觉的最底核单元：
- **归属模态 ($m$)**: 定位所属的物理信道（如灰度拉普拉斯、色彩渐变波等）。
- **空间局域原型核 ($\mathbf{w}^{(n)}$ / Prototype)**: 剥片并独立于原始图像帧被摘除出的抽象张量。

### 3. GmemII 高阶拓扑节点 (SemanticNode)
GmemII 的建立使纯孤立特征组合成为空间物体结构：
- **心锚 (Anchor_ID)**: 标记对象的绝对原点（指向一枚在物体中心的 ModalityNode）。
- **极引力挂载树 (Peripheral Links)**:
  对外围所拾获之点 $k$，建立起相对于其主锚 $anc$ 空间之极坐标关系结构对 $(\Delta\rho_k, \Delta\theta_k)$。以及每个关联所具备的容忍弹性系数 $(\lambda_\rho, \gamma_\theta)$。

---

## 二、 动态投影与空间回放网络 (Gpos: Dynamic Projection Network)

Gpos 充当检索枢纽库。负责将孤立提取的特征张量 $\mathbf{w}$，放入不断变幻的输入视野流（多模矩阵字典 $X_{subspaces}$）中进行空间卷积找寻响度点。

### 1. 模版通道路由式投影 (L1: Modality Routing Projection)
仅当 GmemI 节点满足准入门限（处于 ACTIVE，并且没有在 REFRACTORY 态被隔离）且具有足月起充势量时：
将目标节点 $n$ 的核 $\mathbf{w}_n$ 对入所属通道源 $X_m$，并结合人工专家经验预置权重 $W_{mod}(m)$：
$$S_n(\mathbf{q}) = W_{mod}(m) \cdot \Big[ \mathbf{w}_n \ast X_{m}(\mathbf{q}) \Big]$$
上述式中 $\ast$ 为在局域 C 通道上之点积等价汇和，在整幅空间维度上表现为全局空间相似响应图（Heatmap）。

*(补注：`l1_footprint_projection` 不受限源节点的平静态校验，只要存在激活迹象即强行向空间播撒投射，用于下放基盘之“足迹期望或厌恶”)*

### 2. 拓扑同胚变换与结构共振网络 (L2: GposII Structure Synthesis)
面向在当前视野找回以往记忆拓扑网络（识别复现物体）。以锚为中心对所有挂外源启动一次强行同构核准。

**全息极坐标重采样 (Log-Polar Mesh Sampling)**
搜寻全图中最契合的锚点候补中心 $\mathbf{p}_{anc_{i}} = (x_c, y_c)$。
在该点建立局部的对数极坐标网格变换：
$$r \leftarrow \log(\rho), \quad \theta \leftarrow \arctan\left(\frac{y}{x}\right)$$
把此时刻下所有从属周围节点产生的 L1 特征响应场 $S_k(\mathbf{q})$ 进行张量采样剥离扭曲，转化为以核心锚为参照坐标的高维特征圆环面响应图 $S_{LP}^{(k)}$。

**广义距离变换融合惩罚 (Generalized Distance Transform Approximation)**
各个外围节点根据记录记忆历史之中心位移理想点 $(\Delta \rho_k, \Delta \theta_k)$ 并结合配置好的弹性宽容系数度，利用平滑卷积内核执行粗化弥散逼近：
$$D^{(k)} = \mathcal{G}_{\text{blur}} \Big( S_{LP}^{(k)}(\rho - \Delta\rho_k, \theta - \Delta\theta_k) \Big)$$
*利用卷积泛化出在偏离原记忆拓扑形状位置处响应的热力学衰减斜坡。*

**宏观结构共振坍缩 (Total Peak Resonance & Saccade Yield)**
汇聚出物体拓扑复燃结构图，其最高山峰指出此帧变形与偏差参数位姿：
$$D_{sum}(\rho_{grid}, \theta_{grid}) = \sum_{k} D^{(k)}(\rho_{grid}, \theta_{grid})$$
寻找 $\operatorname{argmax}(D_{sum})$。其解除了印证被寻找主体于视野的可靠存在（Score 置信评分叠加）外，坐标偏移直接解译了主体此刻相较早前发生了何种扭转与伸缩尺度（$\theta_{star}, s_{star}$）。

---

## 三、 算法执行流与交互动力机制 (Execution Flow & Interaction Dynamics)

数据池 Gmem 等待激活，检索网 Gpos 提供空间映射引擎，而连接二者的核心动力即在于 **InterestOptimizer 空间动力学优化算子**与 **Controller 宏微观双轨眼跳伺服状态机**。两者的持续交火涌现出视觉注意力的自发转移路径。

### 1. 异构底层特征基场预处理 (Static Base Map Construction)
每输入一帧新图像体系，首先对其多通道进行属性解耦与过滤：
- **强度引力场** (STRENGTH): 纯振幅提取 $I_{str}(\mathbf{q}) = \sum w_{c} X_{str,c}(\mathbf{q})$
- **表面期望场** (CONTINUITY_SURFACE): 各向同性连续度平滑评估 $val_{surf}(\mathbf{q}) = \exp(-\frac{\|\nabla X_{surf,c}(\mathbf{q})\|^2}{2\sigma_{surf}^2})$
- **线域期望场** (CONTINUITY_TRACE): 求取局域局部张量场一致相干态，计算矢量模长的平滑衰减，聚合出 $I_{trace}(\mathbf{q})$。
融合构建出静态全视角起跳基盘：$I_{base}(\mathbf{q}) = I_{str}(\mathbf{q}) + w_s I_{surf}(\mathbf{q}) + w_t I_{trace}(\mathbf{q})$。

### 2. 空心化促进与绝压陨石坑双模空间调制场 (Spatial Activation Footprint)
眼跳扫描进程中，Gmem 记忆池中正在起振的节点将自身内生神经元振幅 $A_n(t)$ 投射回全图，演化为截然不同的动态阻滞空间足迹 $I_{spatial}$。结合 Gpos 供给的空间相似回响盘 $M_{resp}^{(n)}(\mathbf{q})$，利用绝对值构建出代数统一的双高斯投影调和：
$$I_{spatial}^{(n)}(\mathbf{q}) = M_{resp}^{(n)}(\mathbf{q}) \cdot \Big( A_n(t) \cdot G_{local}(\mathbf{q}, \mathbf{p}_t) - |A_n(t)| \cdot G_{foveal}(\mathbf{q}, \mathbf{p}_t) \Big)$$
- **兴奋游走期 (ACTIVE, $A_n > 0$)**: 场表现为高斯差分（DoG 空心化促进场）。原心靶点处收益抵消趋 0，但在边外围产生诱导晕环，迫使系统自然向邻近未探索的同质介质滑步游走，断绝高分原地横跳死翘。
- **疲劳不应期 (REFRACTORY, $A_n = -1$)**: 公式中差号塌缩成叠加。在曾经走过的热区原地砸出极为深不可测的“连贯陨石坑”（Crater Inhibition）并下施外放全局阴影压制。

实时地形算谱将整合计算这片起伏不定的大陆拓扑：
$$I_{map}(\mathbf{q}) = w_{base} I_{base}(\mathbf{q}) + w_{spI} \sum I_{spatialI} + w_{spII} \sum I_{spatialII}$$

### 3. 微观纯化拓扑掩码的涌现 (Topologic Semantic Mask Diffusion)
在建立 GmemII 锚点 (Anchor) 初期，基于纯提取出来的 `CONTINUITY_SURFACE` 类型节点以防范高频噪声导致的结构错乱，生成用于约束视野流的软拓扑掩码。
**通透率介质导通阻力 $P_{map}$**: 用纯面 Gpos 相似性 $M_{resp\_con}$，叠加 $I_{str}$ 获取之物理锐利物切边作流控阻墙阻断连通性：
$$P_{map}(\mathbf{q}) = M_{resp\_con}(\mathbf{q}) \cdot \operatorname{Sigmoid}(1 - \lambda \cdot I_{str}(\mathbf{q}))$$
**张量蔓延扩散引擎 (Max-Pool Tensor Dilation)**:
设定注视点原初星火 $V_0(\mathbf{p}_{macro}) = 1.0$。基于内置算子持续快速膨胀：
$$V_{t+1} = \text{MaxPool2d}(V_t) \odot P_{map}$$
直至完成预期探索直径内，截定锐化：$M_{semantic}(\mathbf{q}) = \operatorname{Sigmoid}(\gamma \cdot V_T(\mathbf{q}) - \delta)$

### 4. Controller 双极流状态机伺服循环 (Macro / Micro Saccade Control)

#### 4.1 系统依凭底层场形动力构建一套全自动的扫描、探视再遗忘的高级流转。

- **第一极 (Macro-Saccade - 全域猎捕跃迁)**:
  系统处于清空无拘束态（$M_{semantic} = null$）。视界以最原始和最活跃之 $I_{map}$ 制高点驱动进行大跨度飞腾跳跑 $\mathbf{p}_{macro}$。
  落地时执行严谨审查准入录入 GmemI（对于门限值低的碎屑点直接废弃屏蔽不记录）。创建建立 GmemII 结构锚 Anchor。随后即刻通过上述过程唤起拓扑掩码涌现过程。

- **第二极 (Micro-Saccade - 锁区贪婪挖掘)**:
  辖区光圈出现。注意力兴趣被粗暴一刀切绝于禁区外侧：$I_{masked} = I_{map} \odot M_{semantic}$。
  眼跳罚距迅速坍缩变为微距控制圈 $\sigma_{micro}$，确保眼跳链只能牢牢绑定在辖区身躯表皮及边缘上走钢丝。它在限制辖区内对连续暴露并挑出之最抢眼目标 $I_{masked\_max}$ 重复吞食，建立外围关联极元节点置入 Anchor 中附庸挂载。

- **第三终结 (Breakout - 辖区榨衰破溃)**:
  高分节点看一个死一个。满坑满谷全是 REFRACTORY 负压陨石坑的掩模图内部全面崩坏坠崖。
  当全局探顶评估 $\max(I_{masked}) < \epsilon$ 时，确认此物体区域被解构收榨彻底完成。
  当即销毁清除软光圈掩码矩阵 $M_{semantic} \to \text{None}$ 及抹除语义对象专有锁定态。机器解除罚步脚镣，视野重见天日，在辖区外茫茫深山未知迷雾的远核再次起跳复归 Macro Saccade 猎寻环节。

#### 4.2 眼跳驱动策略

1. REVIEW 状态（记忆验证阶段）
*   **驱动目标**：对已有记忆特征（`interest_map`）高响应区域进行复查。
*   **计算模型**：注视点 $p_{next}$ 取决于记忆兴趣值与空间抑制的结合。
    $$p_{next} = \arg\max \left[ I_{map}(\mathbf{q}) - \alpha \cdot IoR(\mathbf{q}) - \gamma \cdot Dist(\mathbf{p}_{curr}, \mathbf{q}) \right]$$
*   **IoR演化**：全局抑制图 $IoR(\mathbf{q})$ 每次注视后在当前点 $\mathbf{p}_{curr}$ 进行高斯叠加并自然衰减，强制打破多次定焦死锁。

2. LEARN_MACRO 状态（宏观锚点搜索阶段）
*   **驱动目标**：寻找视觉中大面积连续平坦色块区域，并**严格规避已经探索建构的物体（语义特征团）辖区**。
*   **计算模型**：
    原采用历史空间的累积抑制 $\sum I_{spatial}^{II}$ 易造成时序衰减或重置，导致抑制失效，引发系统在探索上的周期性震荡（如11步循环死锁）。现调整为直接获取高阶位置图网络（GposII）对当前视野的真实空间位置响应图 $M_{GposII}$ 作为绝对掩码约束。
    $$I_{macro}(\mathbf{q}) = I_{con1}(\mathbf{q}) \odot \left(1 - \operatorname{Norm}(M_{GposII}(\mathbf{q})) \right)$$
    利用形态学空间低通滤波 $K_{lowpass}$ 进行平滑以剔除高频孤立噪声：
    $$p_{next} = \arg\max \left( I_{macro} * K_{lowpass} \right)$$
    以此确保宏观观察点准确降落在未被建构认知区域的大面积连续面状色块的几何中心。

3. LEARN_MICRO 状态（微观特征精细扫描阶段）
*   **驱动目标**：在软掩模 $M_{semantic}$ 限制下的辖区内深度扫描，并实行与该视野面积特征自适应的眼跳约束。
*   **计算模型**：
    回归无偏的底层特征，且仅对软掩模辖区激发。
    $$p_{next} = \arg\max \left( S_{base}(\mathbf{q}) \odot M_{semantic}(\mathbf{q}) \odot (1 - M_{aversion}(\mathbf{q})) \right)$$
*   **动态扫描步长（自适应 $\sigma_{fovea}$）与厌恶足迹机制**：
    固定步长面对比例悬殊的掩码时，极易造成局部遍历过载或边缘越界。需引入几何面积向一维跨度投射的动态调节因子 $\sigma_{fovea}$（映射当前微观扫视的步幅与抑制半径）：
    计算语义掩码面积积分 $A_{semantic} = \sum M_{semantic}(\mathbf{q})$
    进行量纲对齐并缩放生成步幅：
    $$\sigma_{fovea} = \eta \cdot \sqrt{A_{semantic}} + \epsilon_{base}$$
    （其中 $\eta$ 为缩放常数，$\epsilon_{base}$ 确保基础解析限度）
    每次产生新注视点 $\mathbf{p}_i$ 后，使用该自适应 $\sigma_{fovea}$ 在周围施加抑制分布：
    $$M_{aversion}^{(t+1)}(\mathbf{q}) = \max \left( M_{aversion}^{(t)}(\mathbf{q}) , \exp \left( -\frac{||\mathbf{q} - \mathbf{p}_i||^2}{2\sigma_{fovea}^2} \right) \right)$$
*   **退出条件计算**：每次眼跳检查自适应厌恶足迹对当前语义结构的覆盖率指标 $\rho$：
    $$\rho = \frac{\sum (M_{aversion} \odot M_{semantic})}{A_{semantic}}$$
    若 $\rho > \theta_{exit}$ （通常设为 $0.85$ 左右），表示该面状区内信息扫描已达期望水平，触发状态机无条件退出至 `LEARN_MACRO`。