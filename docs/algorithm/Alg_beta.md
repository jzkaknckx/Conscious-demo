# 视觉感知图网络算法探讨与重构指导 (Algorithm Beta)

> **文档说明**：本文档为先行测试版算法探讨库。用于梳理在实践调试中发现的架构缺陷，并制定尚未合入正式 `Algorithm.md` 的前沿修复与规范化草案。

---

## 探讨一：色彩特征原型的替换 (Hue $\to$ RGB) **已完成，等待指令移入Algorithm_gmp.md**

### 1. 现有方案缺陷分析
在以往的代码中，系统使用单一通道的 `hue`（色相）特征作为大面积二维连续面（如天空、桌面）的特征原型并寻找相似区域。这一设计在真实图像环境下暴露出严重不足：
*   **灰度盲区缺失**：黑、白、灰等无彩色系在数学上其 `hue` 值是无意义的噪声或被强制置零。若仅依赖 `hue`，系统完全丧失了对无彩色块的区分能力。
*   **明暗分辨不足**：`hue` 剥离了亮度（Value/Lightness）和饱和度（Saturation），使得深蓝与浅蓝、亮红与暗红被错误地视为具有极高相似度的同一平面，导致区域边界溢出，无法根据光影和材质明暗切分物体。

### 2. 替代方案设计
**结论：直接使用 `RGB`（或 `CIELab`）三通道特征全面替换原有的单通道 `hue`，是非常合理且必要的。**

*   **模态定义升级**：建立全新的 `Modality.COLOR` 模态，其特征张量形如 $\mathbf{X}_{color} = [R, G, B]$。
*   **相似度算法变更**：RGB 特征不再是封闭的周期性属性（不适用余弦与三角展开），而是欧几里得空间中的绝对颜色坐标点。其原型匹配（GposI 响应计算）应当摒弃点积，转而采用**高斯径向基函数（RBF 核）**或基于 L2 距离的衰减函数，以精确控制色彩容差半径：
    $$ S_{color}(\mathbf{q}) = \exp\left(-\frac{\|\mathbf{X}_{color}(\mathbf{q}) - \mathbf{W}_{color}\|^2}{2\sigma_{color}^2}\right) $$
*   **收益**：不仅完美解决了黑白灰的追踪问题，高斯 RBF 的平滑特性还能在动态分辨率机制下，对渐变光影提供极好的宽容度。

---

## 探讨二：多模态特征的规范化抽取与写入工作流 (Universal Extraction Pipeline) **已完成，等待指令移入Algorithm_gmp.md**

### 1. 导致 GposI 响应失效的病理诊断
针对调试中发现的问题：“对于 `grad` 模态传来的双通道特征，若第一个通道不满足 `tau_str_gate`，第二个通道满足 `tau_trace_gate`，导致写入的原型呈现 `tensor([0.0, anyvalidfloat])`，进而致使 GposI 余弦相似度计算失效”。

**根本原因**：底层设计违背了“物理属性隔离”的原则。系统将拥有**独立物理意义**和**独立判定门限**的不同特征（如边缘强度与线条连贯度）强行捆绑拼接在同一个张量数组内。在执行一刀切的全通道内积或余弦相似度计算时，由于存在被置零的无效维度（0.0），导致向量模长计算崩溃或内积响应极低。

### 2. 完全解耦的普适性提取与写入规范
为根治此类问题，对所有存在的（如 5 种）模态，必须执行以下一套**“完全物理隔离、独立门控校验、专属相似度度量”**的标准流水线：

#### Step 1: 模态严格拆分与正交分类门控 (Orthogonal Modality Classification & Gating)
严禁拼凑！每种模态必须是单一且纯粹的物理属性表征。为了兼容原有的特征特性，并保证系统的极简与优雅，我们引入**“正交特征分类法”（Orthogonal Feature Classification）**。
将特征变量的分类解耦为两个完全独立的轴（维度）：**计算度量轴（决定如何算相似度）** 与 **空间拓扑轴（决定如何蔓延与探索）**。

**维度一：计算度量轴 (Metric Axis - 核心在于“如何判断两个特征相同”)**
原先“占有度/属性”的二元分类在引入复杂多模态时不够精细。现根据**数学空间度量法则**及**特征在视觉皮层中的表征方式**，将其严密细分为三类：

*   **欧氏绝对坐标 (Euclidean Absolute Point)**：
    *   **代表特征**：RGB 色彩或 CIELab 色彩。
    *   **归类原因（为什么 RGB 不是矢量？）**：对于矢量（如边缘梯度），零向量 $(0,0)$ 代表物理刺激的缺失（即“没有边缘”）。但对于 RGB，坐标 $(0,0,0)$ 并非“没有特征”，而是代表“纯黑”这一确切的物理状态！如果将 RGB 视为矢量并使用点积或余弦计算相似度，那么纯黑 $(0,0,0)$ 与任何颜色的点积均为 $0$，导致黑色区域永远无法匹配。在视觉皮层中，色彩是通过拮抗细胞（红绿、黄蓝、黑白）编码的，色彩更像三维空间中的一个**绝对位置点**。
    *   **度量法则**：相似度仅由两点间的绝对空间距离决定。采用欧氏距离（L2）配合 **高斯 RBF 核** 计算相似度：$S = \exp(-d^2/2\sigma^2)$。
*   **矢量占有度 (Vector Magnitude)**：
    *   **代表特征**：带有强度的空间梯度 $(dx, dy)$。
    *   **归类原因**：对应 V1 皮层中对特定朝向边缘放电的简单细胞。它包含“方向（类别）”与“模长（能量强度）”。
    *   **度量法则**：采用 **带模长权重的点积（Dot Product）**。这使得同方向但不同对比度的边缘依然能产生高度共振。
*   **周期属性 (Periodic Property)**：
    *   **代表特征**：色相角、纯边缘朝向角。
    *   **归类原因**：纯粹的标量，且具备首尾相接的环形闭合特征（如 $359^{\circ}$ 与 $1^{\circ}$ 极度相似）。
    *   **度量法则**：必须先进行 $ [\cos(\theta), \sin(\theta)] $ 正交展开映射到二维连续空间，再通过 **高阶余弦** 计算相似度。

**维度二：空间拓扑轴 (Topology Axis - 对应原边缘/一维/二维)**
此轴决定了 `LEARN_MICRO` 在微观扫视时，能量扩散（Dilation）的连通域法则，以及 GmemII 的边界包裹关系。
*   **突变边界 (Boundary/Edge)**：阻断性质，高频突变。掩码仅沿山脊线（极狭窄单向）蔓延，用于框定二维面。
*   **一维线域 (Trace/1D-Continuous)**：骨架性质。掩码呈一维线状拓扑扩散。
*   **二维面域 (Surface/2D-Continuous)**：平坦性质。掩码在全向二维平面呈水波纹扩张，直到撞击到“突变边界”。

**现有 5 种基础特征的具体双维度归类及处理范式规范：**
1.  **RGB (三通道色彩)**
    *   **计算度量轴 (`feature_types`)**: `Euclidean_Absolute` (欧氏绝对坐标)
    *   **空间拓扑轴 (`feature_attributes`)**: `Surface_2D` (二维面域)
    *   **门控处理范式**: **范式 A (强耦合整体门控)**。由平坦度期望 $I_{surf}$ 作为主导控制标量触发。
2.  **GRAD (双通道: 边缘强度, 边缘朝向)**
    *   **计算度量轴 (`feature_types`)**: `Vector_Magnitude` (矢量占有度)
    *   **空间拓扑轴 (`feature_attributes`)**: `Boundary_Edge` (突变边界)
    *   **门控处理范式**: **范式 A (强耦合整体门控)**。由强度 $c_0 > \tau_{edge}$ 作为主导控制标量触发。
3.  **CURVATURE (三尺度弯曲度)**
    *   **计算度量轴 (`feature_types`)**: `Euclidean_Absolute` (欧氏绝对坐标)
    *   **空间拓扑轴 (`feature_attributes`)**: `Trace_1D` (一维线域)
    *   **门控处理范式**: **范式 B (多尺度独立拆分)**。每个尺度的响应强度分别独立校验门限，拆分为独立的子模态。
4.  **ASPECT_RATIO (多尺度长宽比)**
    *   **计算度量轴 (`feature_types`)**: `Euclidean_Absolute` (欧氏绝对坐标)
    *   **空间拓扑轴 (`feature_attributes`)**: `Surface_2D` (二维面域)
    *   **门控处理范式**: **范式 C (掩码自适应子空间)**。各尺度/视角特征提取时伴随有效性掩码，匹配时动态降维。
5.  **ORIENTATION (多尺度朝向)**
    *   **计算度量轴 (`feature_types`)**: `Periodic_Property` (周期属性 - 需正交展开)
    *   **空间拓扑轴 (`feature_attributes`)**: `Trace_1D` (一维线域) 或 `Boundary_Edge` (突变边界)
    *   **门控处理范式**: **范式 C (掩码自适应子空间)**。局部遮挡导致特定尺度朝向退化时，掩码机制可保障高维余弦匹配鲁棒性。

在注视点 $\mathbf{p}$ 处，系统遍历所有合法模态 $m$，**各自独立进行阈值校验**。若某模态未达其专属阈值门限，**直接丢弃该模态的提取动作**，绝不生成带 0 的补位残废张量。

#### Step 2: 多通道特征的门控容差与专属缓存检索 (Multi-channel Gating & Cache Retrieval)

在执行局部缓存检索之前，必须先解决一个在多模态并发时极为核心的痛点：**当多通道特征（如含有3个尺度的弯曲度，或兼具强度和角度的梯度）在进行阈值校验时，出现“部分通道达标，部分未达标”的情况该如何处理？**
直接将不达标的通道强制置零（产生如 `[0.0, valid_float]` 的残缺张量）是导致内积、余弦等相似度度量全面失效的罪魁祸首。为此，我们将多通道特征的门控逻辑，依据“通道物理依存关系”，严格规范为三大普适化处理范式：

**范式 A：强耦合整体门控 (Holistic Gating)**
*   **适用特征**：**RGB (三通道)**、**GRAD (c0强度, c1朝向)**。
*   **物理逻辑**：各通道之间存在严密的几何或代数绑定。缺失任何一维，整个张量的物理意义都会崩溃（例如：没有边缘强度的“朝向”是虚无的；缺失G通道的RGB颜色是无意义的）。
*   **处理规范**：设立单一的**“主导控制标量（Dominant Scalar）”**。例如 GRAD 的主导标量是 `c0`（边缘强度）；RGB 的主导标量是该区域的“表面平坦度期望”。**规则**：只要主导标量达到门限，则**全通道数据一并完整提取并写入**；若主导标量未达标，则**整个特征张量直接丢弃**。绝不允许对内部某个附属通道进行零值篡改。

**范式 B：多尺度独立拆分 (Independent Sub-modality Splitting)**
*   **适用特征**：**CURVATURE (三尺度弯曲度)** 等由不同尺度卷积核独立生成且具备低维封闭语义的特征。
*   **物理逻辑**：多个通道实际上代表了相同算子在不同感受野（尺度）下**平行且独立**的测算结果。大尺度下的弯曲与小尺度下的平坦并不矛盾。
*   **处理规范**：严格执行“模态物理隔离”。不应将其生硬打包为一个 3 通道张量，而必须在架构上**将其硬拆分为多个完全独立的子模态**（例如实例化为 `CURV_S1`, `CURV_S2`, `CURV_S3`）。**规则**：对每个尺度的通道分别独立执行门限校验。仅有达标的尺度会各自独立实例化为单独的 GmemI 节点，不达标的尺度直接被抛弃。这彻底消除了残缺张量，且各尺度节点将在空间中各自释放引力场，互不干扰。

**范式 C：掩码自适应子空间 (Masked Subspace Matching)**
*   **适用特征**：**ASPECT_RATIO (多尺度长宽比)**、**ORIENTATION (多尺度朝向)** 等高阶复合属性。
*   **物理逻辑**：多通道代表了实体在空间中不同视点、不同频率的并发属性。在遮挡或特定视角下，部分通道可能发生退化或未达标，但保留下来的通道特征依然具有强烈的结构匹配参考价值。
*   **处理规范**：引入**“有效性掩码 (Validity Mask)”**机制。特征提取时，除了保存特征值张量 $\mathbf{X}$ 外，必须强制同步生成等长的布尔掩码数组 $\mathbf{M} = [m_1, m_2, \dots, m_k]$。
    *   **提取侧**：未达标的通道对应掩码置 $0$（该通道的特征值保留原始计算浮点或置 NaN，但不人为写 $0.0$ 破坏数学分布）。
    *   **度量侧**：当调用算子 $S = \text{Sim}(\mathbf{X}, \mathbf{W})$ 进行缓存检索时，系统将动态提取两者的联合掩码交集 $\mathbf{M}_X \land \mathbf{M}_W$。**即：算子会动态降维，只在双方均有效的交集子空间内计算相似度（如降维欧氏距离或正交余弦）**。这确保了局部遮挡或特征残缺不会破坏整体的高维结构鲁棒匹配。

---
**执行缓存检索与去重逻辑**：
经历上述三大范式之一清洗、确认为合法提取的特征张量 $\mathbf{X}_m$（及附属掩码）后，在 GmemI 活跃节点库中进行查重：
1.  **调度专属算子**：依据“计算度量轴”的分类严格分配算法（RGB 调 RBF 核，GRAD 调带模长加权点积，ORIENTATION 调高阶正交余弦等）。
2.  **命中复用**：若算得最高相似度 $S > \theta_{match}$，则**放弃新建**节点，直接提取该历史节点作为此步的观察指针，并流入网络累加该节点的激发势能。

#### Step 3: 原型固化与明确的 GmemI 注册 (Prototype Registration)
若 Step 2 中未命中相似节点，视作新发现：
*   实例化全新 `ModalityNode`，将干净的 $\mathbf{X}_m$ 存为原型 $\mathbf{W}_{new}$。
*   打上明确的模态归属标签：`node.modality = m`（这为后续 GmemII 同源模态过滤提供唯一凭证）。

#### Step 4: GposI 响应图异步派发 (GposI Projection Dispatch)
新节点入列后，立刻基于其 `modality` 标识，调用对应的底层特征矩阵源 $X_m$ 及专属相似度算法，生成其在全图的 GposI 响应热力图 $S_{new}(\mathbf{q})$。
这保证了此后在 `run_step()` 的微观能量扩散或下一次的检索中，该节点立刻具备了引导视线的物理力场。


---

## 探讨：眼动引导机制与图结构边关系的重构 (Saccade Guidance & Hyperedge Relations)

### 一、现有机制的局限与解决方法

#### 1. 现有眼动引导机制的局限性
当前的视线引导策略在面临大块“二维连续特征”（如大面积纯色桌面、天空）时存在严重缺陷。由于全局采用了单一的距离惩罚机制（倾向于选择距离当前注视点近的特征），导致视线被困在平坦的同质化区域内“打转”。

在人类视觉认知中，当视网膜中央凹识别到当前区域是一个巨大的重复/连续面时，眼球不会在内部盲目游走，而是会迅速产生**边界逃逸（Boundary Escape）**眼跳，转向形状边缘、端点、角点或交点，以获取物体的几何轮廓（Gestalt 完形）。因此，MICRO / MACRO 状态机必须引入随特征拓扑属性动态变化的目标引导场。

#### 2. 基于拓扑属性的眼动修正原则
视线目标和距离项应由当前特征拓扑决定：

*   **二维面域 (Surface_2D)**：当视线连续命中相同面域特征节点（如 `Modality.COLOR` 命中缓存）时，触发面域厌倦。此时 GposI 面域响应不应继续把视线困在内部，而应与 Boundary/Grad 响应共同形成边界候选，引导视线向连续面与突变边界的交界线跳跃。
*   **一维线域 (Trace_1D)**：当视线跟踪连续线段时，目标场应沿当前线段切线方向前推，并优先选择端点（Terminators）、角点、交点（Junctions）和高曲率片段。
*   **动态距离调节**：距离衰减系数 $\gamma$ 不能是常数。MACRO 状态应弱化距离项以允许格式塔式大跳；MICRO 初期确认阶段可保留局部约束；当系统进入边界逃逸或轮廓跟随时，距离项应转化为最小跨度、环带或切向带通约束，而不是单纯近距离偏置。

#### 3. 从单一 semantic 耗竭改为拓扑意图场
单纯“边界逃逸”只能解决视线离开同质面域的问题，但不足以完整学习**复杂边缘的大块二维连续特征**。复杂面域至少包含两类互补信息：内部颜色/材质原型 $\mathbf{w}_{surf}$ 与外部轮廓函数 $\partial \Omega$。若仅在面域内部耗竭 `semantic_mask`，系统会重复确认颜色，却无法高效编码轮廓；若仅跃向边界，又会丢失内部颜色统计。

因此，`LEARN_MICRO` 不应被视为一个均质扫描状态，而应拆解为由同一状态机内部调度的若干**拓扑意图 (Intention)**。每个意图只改变眼跳目标场，不改变 Gmem/Gpos 的总体架构：

1.  `SURFACE_CONFIRM`：确认当前面域原型，估计内部颜色均值、方差与稳定性。
2.  `INTERIOR_SAMPLE`：用少量代表点采样内部，避免把大面积同质区域逐点扫完。
3.  `BOUNDARY_SEEK`：从面域内部发起大跨度边界逃逸。
4.  `CONTOUR_TRACE`：沿边界切线方向采样端点、角点、交点与高曲率片段。
5.  `RESUME_OR_EXIT`：根据内部置信度与边界覆盖度决定恢复、挂起或退出到 `LEARN_MACRO`。

#### 4. 面域 MICRO 的 Gpos 响应目标场定义
经过 Gpos 位置相似度审查后，`LEARN_MICRO` 中各意图的候选场不应再写成脱离 Gpos 的抽象兴趣项，而应明确由 Gpos 响应构成。设：

*   $S^{I}_{surf}(\mathbf{q}; n_s)$：Surface/Color 类 GmemI 节点 $n_s$ 的 GposI 响应。
*   $S^{I}_{color}(\mathbf{q}; n_c)$：当前面域颜色原型 $n_c$ 的 GposI 响应；若 $n_s$ 本身即为颜色面节点，可令 $n_c=n_s$。
*   $S^{I}_{edge}(\mathbf{q})$：Boundary/Grad 类边界节点或边缘基场的 GposI 响应。
*   $S^{I}_{corner}(\mathbf{q})$、$S^{I}_{junction}(\mathbf{q})$、$S^{I}_{term}(\mathbf{q})$：由曲率、方向突变、端点等 GmemI 节点投影得到的 GposI 响应。若对应节点尚未稳定形成，可由局部几何算子临时近似，但语义上仍视为待注册的 GposI 候选响应。
*   $S^{II}_{sem}(\mathbf{q})$：同类已学习 GmemII 的结构响应。当前代码尚未可靠实现完整 GposII，因此新对象学习时该项可置零；在 REVIEW 或熟悉对象复现时，该项作为结构先验接入。

当前面域掩码不再是独立经验场，而由 GposI Surface 响应经过连通扩散得到：
$$
M_{\Omega}(\mathbf{q})
=
\operatorname{ConnDiffuse}_{\mathbf{p}_0}
\left(
\operatorname{Norm}\left(S^{I}_{surf}(\mathbf{q};n_s)\right),
\tau_{\Omega}
\right)
$$
其中 $\mathbf{p}_0$ 是宏观落点或当前 GmemII 锚点，$\operatorname{ConnDiffuse}$ 表示以 $\mathbf{p}_0$ 为种子的连通域扩散，仅允许在同源 Surface 响应足够高的区域内传播。

##### 4.1 `SURFACE_CONFIRM` 与 `INTERIOR_SAMPLE`
`SURFACE_CONFIRM` 的目标是确认当前注视仍在同一面域原型中，因此其确认场为：
$$
U_{confirm}(\mathbf{q})
=
M_{\Omega}(\mathbf{q})
\odot
S^{I}_{surf}(\mathbf{q};n_s)
$$

`INTERIOR_SAMPLE` 的目标是用少量代表点学习内部颜色统计。内部候选应同时考虑同源颜色响应、未被现有颜色原型解释的残差响应，以及代表点间隔：
$$
R^{I}_{color}(\mathbf{q})
=
1 -
\max_{n_c \in \mathcal{C}_{\Omega}}
S^{I}_{color}(\mathbf{q};n_c)
$$
$$
U_{in}(\mathbf{q})
=
M_{\Omega}(\mathbf{q})
\odot
\left[
w_c S^{I}_{color}(\mathbf{q};n_c)
+ w_r R^{I}_{color}(\mathbf{q})
+ w_d D_{blue}(\mathbf{q}, \mathcal{P}_{in})
\right]
$$
其中 $\mathcal{C}_{\Omega}$ 是当前面域已注册或正在统计的颜色原型集合，$R^{I}_{color}$ 表示当前 GposI 颜色节点集合解释不了的内部残差。若 $R^{I}_{color}$ 持续形成连通高值区域，则不应继续把它并入同一颜色统计，而应触发子区域分裂或新颜色节点注册。

内部采样停止条件为：
$$
N_{in} \ge N_{min}
\quad \land \quad
\operatorname{Tr}(\Sigma_{color}) < \theta_{color\_var}
\quad \land \quad
\sum M_{\Omega} \odot R^{I}_{color} < \theta_{res}
$$
这保证“大面积纯色”被压缩为颜色均值、协方差、样本数与置信度；同时保留对内部异色斑块或纹理残差的发现能力。

##### 4.2 `BOUNDARY_SEEK`
二维面域的边界候选必须来自 Surface GposI 与 Boundary GposI 的交界，而不是来自纯空间距离项。定义：
$$
B_{\Omega}^{I}(\mathbf{q})
=
\operatorname{Norm}\left(
\left|\nabla M_{\Omega}(\mathbf{q})\right|
\right)
\odot
\operatorname{Norm}\left(
w_e S^{I}_{edge}(\mathbf{q})
+ w_k S^{I}_{corner}(\mathbf{q})
\right)
$$
若已有同类 GmemII 被识别或处于 REVIEW 复现阶段，可加入结构先验：
$$
B_{\Omega}^{Gpos}(\mathbf{q})
=
B_{\Omega}^{I}(\mathbf{q})
+ w_{II} S^{II}_{sem}(\mathbf{q})
$$
其中 $S^{II}_{sem}$ 只能在 GposII 结构位置相似度实现可靠后作为正式项；当前实现阶段可令 $w_{II}=0$。

边界逃逸目标场为：
$$
U_{bd}(\mathbf{q})
=
\left[
w_b B_{\Omega}^{Gpos}(\mathbf{q})
+ w_u U_{unseen}^{bd}(\mathbf{q})
\right]
\odot R_{span}(\mathbf{q};\mathbf{p}_0)
\odot
\left(1-O_{visited}^{bd}(\mathbf{q})\right)
$$
其中 $U_{unseen}^{bd}$ 与 $O_{visited}^{bd}$ 分别表示边界未访问增益和边界已访问抑制；它们作用于 Gpos 边界候选，不再单独制造候选。若需要空间约束，应作为眼跳带通约束叠加到目标场上：
$$
G_{seek}(\mathbf{q})
=
\exp\left(
-\frac{(\|\mathbf{q}-\mathbf{p}_t\|-s_{seek})^2}{2\sigma_{seek}^2}
\right),
\quad
\tilde{U}_{bd}=U_{bd}\odot G_{seek}
$$

##### 4.3 `CONTOUR_TRACE`
当注视点已经进入边界带，系统进入 `CONTOUR_TRACE`。轮廓候选来自边缘、角点、交点、端点等 GposI 响应；若对象已被结构识别，可再叠加 GposII 的边界链预测响应：
$$
S^{Gpos}_{contour}(\mathbf{q})
=
w_e S^{I}_{edge}(\mathbf{q})
+ w_k S^{I}_{corner}(\mathbf{q})
+ w_j S^{I}_{junction}(\mathbf{q})
+ w_t S^{I}_{term}(\mathbf{q})
+ w_{II} S^{II}_{boundary}(\mathbf{q})
$$
其中 $S^{II}_{boundary}$ 表示已学习 GmemII 边界链对当前视野的结构预测；新对象学习或 GposII 尚未可靠实现时令其为零。

设上一段眼跳平滑出的边界切向量为 $\hat{\mathbf{t}}_t$，期望边界步长为 $s_{edge}$，则切向和步长只作为空间约束项：
$$
G_{tan}(\mathbf{q})
=
\exp\left(
-\frac{\angle(\mathbf{q}-\mathbf{p}_t,\hat{\mathbf{t}}_t)^2}{2\sigma_\theta^2}
\right)
\cdot
\exp\left(
-\frac{(\|\mathbf{q}-\mathbf{p}_t\|-s_{edge})^2}{2\sigma_s^2}
\right)
$$
最终轮廓跟随目标场为：
$$
U_{trace}(\mathbf{q})
=
B_{\Omega}^{Gpos}(\mathbf{q})
\odot
S^{Gpos}_{contour}(\mathbf{q})
\odot
G_{tan}(\mathbf{q})
\odot
\left(1-O_{bd}(\mathbf{q})\right)
$$
该式将原先独立的角点、端点和交点增益全部改写为 Gpos 响应项；距离相关项仅限定眼跳步长和切向，不再作为候选来源。

##### 4.4 `RESUME_OR_EXIT`
退出与恢复判定也应使用 Gpos 生成的内部残差和边界覆盖，而不是使用独立经验场。定义边界覆盖率：
$$
C_{bd}
=
\frac{
\sum O_{bd}(\mathbf{q}) \odot B_{\Omega}^{Gpos}(\mathbf{q})
}{
\sum B_{\Omega}^{Gpos}(\mathbf{q}) + \epsilon
}
$$
内部残差率为：
$$
C_{res}
=
\frac{
\sum M_{\Omega}(\mathbf{q}) \odot R^{I}_{color}(\mathbf{q})
}{
\sum M_{\Omega}(\mathbf{q})+\epsilon
}
$$
当 $C_{bd}>\theta_{bd\_cover}$、$C_{res}<\theta_{res}$ 且颜色统计稳定时，当前面域可退出；若 $B_{\Omega}^{Gpos}$ 指向未访问强边界或 $R^{I}_{color}$ 指向未解释内部子区，则进入 `BOUNDARY_SEEK` 或 `INTERIOR_SAMPLE`。

#### 5. MICRO / MACRO 切换逻辑修正
原有 `LEARN_MICRO -> LEARN_MACRO` 的“模态突变退出”需要细分。对于二维面域，触碰边界不应立即视为离开当前语义对象，而应首先视为该对象的**预期边界发现**。

推荐状态转移：

1.  `LEARN_MACRO -> LEARN_MICRO`：落点位于未探索连续面域中心，创建以 Surface_2D 节点为锚的 GmemII。
2.  `SURFACE_CONFIRM -> INTERIOR_SAMPLE`：连续命中同一 Surface_2D 原型，且 $A_{\Omega}=\sum M_{\Omega}$ 或 $S_{span}$ 超过小物体阈值。
3.  `INTERIOR_SAMPLE -> BOUNDARY_SEEK`：内部颜色统计稳定，或重复命中同一原型次数超过 $\theta_{repeat}$。
4.  `BOUNDARY_SEEK -> CONTOUR_TRACE`：`BOUNDARY_SEEK` 只能产生边界候选点 $\mathbf{p}^{*}$，不应仅凭全局候选场峰值立即切换。系统应在实际观察 $\mathbf{p}_{t+1}=\mathbf{p}^{*}$ 后，根据局部边界命中置信度和前向轮廓支持进入 `CONTOUR_TRACE`。
5.  `CONTOUR_TRACE -> BOUNDARY_SEEK`：当前边界段到达局部终点但边界覆盖率不足，重新选择未访问边界候选。
6.  `CONTOUR_TRACE -> INTERIOR_SAMPLE`：边界或内部 GposI 残差提示存在多颜色/纹理子区，即 $C_{res}>\theta_{res}$，回到内部做子区域分裂。
7.  `LEARN_MICRO -> LEARN_MACRO`：同时满足内部稳定度与边界覆盖度：
$$
C_{done} =
\left[\operatorname{Tr}(\Sigma_{color}) < \theta_{color\_var}\right]
\land
\left[C_{bd} > \theta_{bd\_cover}\right]
\land
\left[C_{res} < \theta_{res}\right]
$$
8.  `挂起/恢复`：若边界外侧出现强异质特征或另一个对象，当前面域 GmemII 标记为 `is_completed = false` 并挂起；学习完外侧子结构后恢复该面域的未访问边界段。

#### 6. 调试问题：`BOUNDARY_SEEK` 死锁与 `CONTOUR_TRACE` 早退
工程实现中出现两个相关现象：其一，约第 10 步后意图长期停留在 `BOUNDARY_SEEK`，且 `_decide_next_saccade_micro_by_intention()` 中的 `drive` 峰值总是落在高边缘响应附近；其二，前期虽两次进入 `CONTOUR_TRACE`，但分别在观察 1 步和 2 步后退出，同时 GmemI 节点长期只有第一个色相节点和第一个边缘节点。该现象说明当前实现把“哪里存在强边缘候选”和“视线已经命中并可沿边界追踪”混为同一判据，并且局部特征匹配把后续边界观察过度合并到第一个边缘原型。

##### 6.1 不应使用 `drive.max()` 作为状态切换证据
`BOUNDARY_SEEK` 中的 `drive` 是一个全局候选场：
$$
\tilde{U}_{bd}(\mathbf{q})
=
U_{bd}(\mathbf{q})
\odot
G_{seek}(\mathbf{q})
\odot
\left(1-O_{visited}^{bd}(\mathbf{q})\right)
$$
它回答的是“下一眼最值得跳到哪里”，而不是“当前视线是否已经位于可追踪边界”。因此，`drive.max()>\theta_{boundary}` 只表示图上某处仍有强边界候选；当边缘响应很强时，该条件天然成立，容易把 `BOUNDARY_SEEK` 过早改成 `CONTOUR_TRACE`。随后 `CONTOUR_TRACE` 又用 `drive.max()<\theta_{boundary}` 判断退出，若切向门控、已访问抑制或边界访问半径过大导致前向候选被压低，系统会在一两步内退回 `BOUNDARY_SEEK`。

应将候选场强度与局部命中证据分离。设 $\mathcal{F}(\mathbf{p})$ 为以注视点为中心的中央凹窗口，定义边界命中置信度：
$$
C_{hit}(\mathbf{p})
=
\frac{
\sum_{\mathbf{q}\in\mathcal{F}(\mathbf{p})}
B_{\Omega}^{Gpos}(\mathbf{q})
}{
|\mathcal{F}(\mathbf{p})|+\epsilon
}
$$
定义前向轮廓支持：
$$
C_{fwd}(\mathbf{p},\hat{\mathbf{t}})
=
\max_{\mathbf{q}}
\left[
B_{\Omega}^{Gpos}(\mathbf{q})
\odot
S^{Gpos}_{contour}(\mathbf{q})
\odot
G_{tan}(\mathbf{q};\mathbf{p},\hat{\mathbf{t}})
\right]
$$
其中 $C_{hit}$ 只能由当前注视点或刚完成眼跳后的实际落点计算；$C_{fwd}$ 只用于判断当前边界点是否存在可追踪方向。推荐切换条件为：
$$
C_{hit}(\mathbf{p}_{t})>\theta_{hit}^{enter}
\quad\land\quad
C_{fwd}(\mathbf{p}_{t},\hat{\mathbf{t}}_0)>\theta_{fwd}^{enter}
$$
而不是 $\max_{\mathbf{q}}\tilde{U}_{bd}(\mathbf{q})>\theta_{boundary}$。

更稳健的状态流程为：

1.  `BOUNDARY_SEEK` 计算 $\tilde{U}_{bd}$ 并选择 $\mathbf{p}^{*}=\arg\max \tilde{U}_{bd}$，只记录 `pending_boundary_target = p*`。
2.  眼跳到 $\mathbf{p}^{*}$ 并完成一次局部观察后，计算 $C_{hit}(\mathbf{p}_{t})$。
3.  若 $C_{hit}$ 和前向支持均达标，则进入 `CONTOUR_TRACE`；否则仍留在 `BOUNDARY_SEEK`，并将该落点写入短时 `boundary_rejected` 或失败访问图，避免下一步继续选择同一个强峰。
4.  `CONTOUR_TRACE` 退出应使用迟滞条件：$\theta_{hit}^{enter}>\theta_{hit}^{exit}$、$\theta_{fwd}^{enter}>\theta_{fwd}^{exit}$，并要求 `trace_age >= T_trace_min` 后才允许退出。

因此，`CONTOUR_TRACE -> BOUNDARY_SEEK` 的判据应改为：
$$
trace\_age\ge T_{min}
\quad\land\quad
C_{fwd}(\mathbf{p}_{t},\hat{\mathbf{t}}_t)<\theta_{fwd}^{exit}
\quad\land\quad
C_{hit}(\mathbf{p}_{t})<\theta_{hit}^{exit}
$$
若当前点是端点、角点或交点，则不应直接视为失败退出，而应先记录相应边界事件，再由边界覆盖率决定继续沿另一方向追踪还是重新 `BOUNDARY_SEEK`。

##### 6.2 切向初始化与访问抑制需要延迟
当前实现中，`BOUNDARY_SEEK` 若在生成候选场时直接把意图改成 `CONTOUR_TRACE`，后续 `_decide_next_saccade_micro_by_intention()` 会用“从面域内部跳向边界”的眼跳向量更新 `last_boundary_tangent`。该向量更接近面域法向或逃逸方向，不是边界切向，容易使下一步 `G_{tan}` 指向错误方向，造成 `CONTOUR_TRACE` 早退。

切向初始化应来自边界局部几何，而不是来自边界逃逸眼跳。可采用以下任一方式：

*   若已有梯度或边缘方向通道，令 $\hat{\mathbf{t}}_0$ 为局部边缘方向的圆均值。
*   若只有面域掩码，先估计法向 $\hat{\mathbf{n}}=\operatorname{Norm}(\nabla M_{\Omega})$，再令 $\hat{\mathbf{t}}_0=R_{90}(\hat{\mathbf{n}})$。
*   若方向不稳定，先进入一个短暂 `TRACE_INIT` 子阶段，在同一边界带中选择两个相邻命中点后再平滑出 $\hat{\mathbf{t}}_t$。

此外，`boundary_visited` 不宜在计算下一步追踪目标前过早、过宽地抑制当前边界邻域。推荐顺序为：先根据未抑制或弱抑制的局部边界场计算 $C_{hit}$ 与 $C_{fwd}$，再在决定下一眼之后标记已访问边界。访问抑制应主要抑制身后或已覆盖弧段，而不是抑制当前点前方的切向邻域；必要时将访问图分为 `boundary_visited_for_coverage` 与 `boundary_suppression_for_target`，前者用于覆盖率统计，后者用于目标选择。

##### 6.3 `_extract_and_match_local_features()` 的相似度需要拓扑上下文
GmemI 节点长期只有两个，说明当前 `_extract_and_match_local_features()` 的相似度检测把后续边界观察全部匹配到了第一次看到的边缘节点。对于颜色面域，这种原型合并是合理的；但对于复杂轮廓学习，仅用同一模态的局部特征向量相似度会丢失“这是同一边缘原型在不同轮廓位置、不同方向、不同曲率或不同边界角色上的一次观察”。

需要先区分两类对象：

*   **特征原型节点**：表示“这种颜色”或“这种边缘响应类型”，可以被大量复用。
*   **局部观察实例 / 关系边实例**：表示某次中央凹观察在当前对象结构中的位置、方向、角色和与其他观察的位移关系，即使复用了同一个 GmemI 原型，也必须在 GmemII 的多重边或超边中新增一条观察记录。

因此，不应把 “GmemI 节点数没有增加” 直接等价为 “没有学到新边界”。最低限度的实现要求是：当边缘原型被匹配复用时，仍要向当前 GmemII 写入新的 observation relation，记录：
$$
(\text{prototype\_id},\ \Delta \mathbf{p},\ \hat{\mathbf{t}},\ \kappa,\ \text{role},\ \text{intention},\ t)
$$
其中 $\Delta \mathbf{p}$ 是相对当前 GmemII 锚点或上一边界观察的位移，$\hat{\mathbf{t}}$ 是局部切向，$\kappa$ 是曲率或方向变化，`role` 可取 `surface_sample`、`boundary_hit`、`corner`、`junction`、`terminator` 等。

若工程上仍希望 GmemI 产生可区分的边界地标节点，则相似度应改为拓扑条件化的复合相似度：
$$
S_{total}
=
S_{feat}^{\alpha}
\cdot
S_{geom}^{\beta}
\cdot
S_{role}^{\gamma}
$$
其中：
$$
S_{geom}
=
\exp\left(-\frac{\Delta\theta^2}{2\sigma_{\theta}^2}\right)
\cdot
\exp\left(-\frac{(\kappa-\kappa_i)^2}{2\sigma_{\kappa}^2}\right)
$$
$S_{role}$ 表示边界命中、角点、交点、端点等拓扑角色是否一致。对于 `Surface_2D`，可保持较高合并倾向；对于 `Boundary_Edge` 与 `Trace_1D`，尤其在 `BOUNDARY_SEEK` / `CONTOUR_TRACE` 中，应提高角色和几何上下文权重，或采用更高的匹配阈值。若当前边缘特征向量只有强度而缺少方向、曲率、端点/交点响应，则无论阈值如何调整都会倾向于过度合并；应把局部方向、曲率和拓扑角色作为特征通道、mask 通道或 relation 属性接入。

推荐的匹配策略为：

1.  `Surface_2D`：优先合并到颜色/材质原型节点，通过均值、方差和样本数编码内部统计。
2.  `Boundary_Edge`：允许复用边缘原型，但每次命中边界都必须新增关系观察；当方向、曲率或角色差异超过阈值时，创建新的边界地标节点。
3.  `Trace_1D`：相似度必须包含局部切向连续性；若方向突变、端点或交点响应显著，应强制生成事件节点或事件关系。
4.  `CONTOUR_TRACE` 期间：即使 `S_feat` 很高，只要 $S_{geom}$ 或 $S_{role}$ 低于阈值，也不应把观察静默合并为同一个普通边缘节点。

这样可以避免两个极端：既不会为大面积同质颜色生成大量重复 GmemI 节点，也不会把复杂边界压缩成第一个边缘节点而无法形成轮廓结构。

##### 6.4 可接受复杂度的实现建议
上述修正不要求引入高复杂度全局规划。可在现有实现中增加少量局部状态和局部算子：

*   新增 `pending_boundary_target`、`trace_age`、`boundary_rejected`、`trace_init_points` 等 MICRO 上下文变量。
*   用中央凹窗口均值或小卷积计算 $C_{hit}$，用当前 `G_{tan}` 候选带中的最大值或 top-k 均值计算 $C_{fwd}$。
*   用迟滞阈值替代单一 `theta_boundary`：`theta_hit_enter`、`theta_hit_exit`、`theta_fwd_enter`、`theta_fwd_exit`。
*   将边界访问图拆成覆盖率统计与目标抑制两种用途，避免为了覆盖率而破坏下一步前向追踪。
*   在 `_extract_and_match_local_features()` 中加入 topology-aware 分支：颜色面域维持强合并，边界/线域引入方向、曲率和角色上下文；无论是否新建 GmemI 节点，都向当前 GmemII 记录观察关系。

若只做最小修复，优先级应为：先移除 `BOUNDARY_SEEK` 中基于 `drive.max()` 的即时切换，改为“候选点观察后命中”；再给 `CONTOUR_TRACE` 加入最小追踪步数与迟滞退出；最后将边界原型匹配与关系观察记录解耦。

#### 7. 距离项的普适调度原则
距离项 $\gamma$ 不应作为全局常数，而应由当前拓扑意图决定：

| 状态/意图 | 距离项形式 | 行为含义 |
| --- | --- | --- |
| `REVIEW` | $-\gamma\|\mathbf{q}-\mathbf{p}_t\|$ + IoR | 稳定复查，避免无谓大跳 |
| `LEARN_MACRO` | 极弱距离项 + 已探索语义抑制 | 搜索新对象中心，允许格式塔大跳 |
| `SURFACE_CONFIRM` | 小范围局部约束 | 确认当前面域原型 |
| `INTERIOR_SAMPLE` | 蓝噪声/最远点间隔 | 少量代表点覆盖内部 |
| `BOUNDARY_SEEK` | 最小跨度或环带约束 | 从同质内部跳向边缘 |
| `CONTOUR_TRACE` | 切线方向 + 带通步长 | 沿轮廓前进，优先端点/角点 |
| `Trace_1D` 线域 | 端点牵引 + 切向预测 | 沿线段找端点、交点、分叉 |

这一调度保留距离项作为普适机制，但将“近距离抑制/偏置”推广为“由当前拓扑意图定义的空间约束”。复杂度主要来自若干卷积、池化、形态学梯度和少量状态变量，能够在现有 `InterestOptimizer` 与 `Controller` 框架内实现。

---

## 补充：探讨二修订版 - 多模态通道依赖、门控与相似度统一规范

本节用于覆盖前文“探讨二：多模态特征的规范化抽取与写入工作流”中过于粗糙的通道分类。原章节提出“欧氏绝对坐标 / 矢量占有度 / 周期属性”三类相似度轴是合理的，但仍存在两个实现层面的缺陷：

1.  分类粒度停留在 `mod_id` 层，而不是子通道层。`grad` 的强度、角度、`gradx`、`grady` 物理角色不同，不能被同一个 `Vector_Magnitude` 算子整体点积。
2.  门控掩码与相似度掩码混用。某些通道只应决定“该特征是否有效”，不应参与“该特征与原型是否相似”的数值计算。

因此，新的规范不再把一个模态张量直接交给单一相似度函数，而是先建立**通道契约 (Channel Contract)**。每个子通道必须声明：

*   `role`：`gate_only`、`sim_only`、`gate_and_sim` 或 `context_only`。
*   `metric`：`Euclidean_Absolute`、`Vector_Occupancy`、`Periodic_Property` 或 `None`。
*   `parent_gate`：该通道有效所依赖的父门控。
*   `period`：周期通道的周期，非周期通道为空。
*   `sigma / gamma / weight / theta_match`：该通道或通道组的独立检索参数。

### 1. 五类输入模态的修订契约

输入仍保持五类模态：`grad`、`RGB`、`Curv`、`aps`、`ori`。其中 `grad` 从原双通道改为四通道：

| mod_id | 模态 | 子通道 | role | metric | topology | 门控依赖 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `grad` | `c0=abs_strength` | `gate_only` | `None` | `Boundary_Edge` | 原始强度门控 |
| 0 | `grad` | `c1=theta=atan2(grady, gradx)` | `gate_only` / `context_only` | `None` | `Boundary_Edge` | 受 `c0` 强度约束，用于一维 coherence 与切向初始化 |
| 0 | `grad` | `c2=gradx` | `sim_only` | `Vector_Occupancy` | `Boundary_Edge` | `c0` 强度达标且 `c1` 局部 coherence 达标 |
| 0 | `grad` | `c3=grady` | `sim_only` | `Vector_Occupancy` | `Boundary_Edge` | 同 `c2` |
| 1 | `RGB` | `c0=R,c1=G,c2=B` | `gate_and_sim` | `Euclidean_Absolute` | `Surface_2D` | RGB 三通道整体受二维平滑度门控 |
| 2 | `Curv` | 多尺度曲率 | `gate_and_sim` | `Euclidean_Absolute` | `Trace_1D` / `Boundary_Event` | 受边界/线域父门控与本尺度曲率置信度门控 |
| 3 | `aps` | 多尺度长宽比/形状比例 | `gate_and_sim` | `Euclidean_Absolute` | `Surface_2D` / `Entity_Shape` | 受连通域面积、占有度和尺度支持度门控 |
| 4 | `ori` | 多尺度朝向 | `gate_and_sim` | `Periodic_Property` | `Trace_1D` / `Boundary_Edge` | 受同尺度方向 coherence 与父边界/线域支持门控 |

`grad.c1` 与 `ori` 的职责必须分开：`grad.c1` 是从局部梯度即时得到的门控和几何上下文，不直接作为 GmemI 相似度维度；`ori` 是经过尺度汇聚、稳定性筛选后的显式周期特征，才参与周期相似度检索。

### 2. 门控掩码与相似度掩码分离

每次局部抽取不再只生成一个 `valid_mask`，而应至少生成三类掩码或等价结构：

$$
M^{gate}_m,\quad M^{sim}_m,\quad M^{ctx}_m
$$

*   $M^{gate}$ 决定该模态本次是否允许写入或检索。
*   $M^{sim}$ 决定哪些通道参与相似度计算。
*   $M^{ctx}$ 保存只供状态机、关系边或拓扑角色使用的上下文通道。

任一 `gate_only` 通道即使数值有效，也不得进入 $M^{sim}$。例如 `grad.c0` 与 `grad.c1` 可被写入 `gate_context`，用于更新节点置信度、边界命中强度、切向初始化或 relation 元数据，但不能直接参与 `SimilarityEngine.calculate_similarity()` 的向量点积。

推荐局部抽取流程：

1.  先计算所有父门控：强度、二维平滑度、一维 coherence、尺度支持度、连通域支持度。
2.  若父门控不通过，直接丢弃该模态本次观察，不生成零值补位原型。
3.  若父门控通过，只把 `sim_only` 与 `gate_and_sim` 通道加入 $M^{sim}$。
4.  相似度检索只在 $M^{sim}_X \land M^{sim}_W$ 的交集上进行；交集不足 `min_valid_sim_channels` 时直接判为不匹配。
5.  GmemI 原型更新只更新相似度通道；门控与上下文通道以统计量形式保存在节点或 relation 元数据中。

### 3. 完备的子模态门控方案

#### 3.1 `grad` 门控

设：
$$
g(\mathbf{p})=c_0(\mathbf{p})=\sqrt{g_x^2+g_y^2},
\quad
\theta(\mathbf{p})=c_1(\mathbf{p})=\operatorname{atan2}(g_y,g_x)
$$

强度门控：
$$
G_{str}^{grad}(\mathbf{p})
=
\left[g(\mathbf{p})>\tau_{grad\_str}\right]
$$

一维方向 coherence 应使用强度加权的圆统计，而不是直接对角度做线性平滑：
$$
R_{grad}(\mathbf{p})
=
\frac{
\left|
\sum_{\mathbf{q}\in\mathcal{N}(\mathbf{p})}
g(\mathbf{q}) \exp(i k_{grad}\theta(\mathbf{q}))
\right|
}{
\sum_{\mathbf{q}\in\mathcal{N}(\mathbf{p})} g(\mathbf{q})+\epsilon
}
$$

其中 $k_{grad}=1$ 表示保留梯度极性，$k_{grad}=2$ 表示只关心边缘轴向、不区分黑白极性。用于对象轮廓学习时，默认应采用 $k_{grad}=2$，否则同一轮廓的明暗反转会被误判为不同边界。

最终门控：
$$
G^{grad}(\mathbf{p})
=
G_{str}^{grad}(\mathbf{p})
\land
\left[R_{grad}(\mathbf{p})>\tau_{grad\_coh}\right]
$$

只有 $G^{grad}=1$ 时，`c2=gradx` 与 `c3=grady` 才进入 $M^{sim}$。`c0` 和 `c1` 只进入 $M^{gate}$ 或 $M^{ctx}$。

#### 3.2 `RGB` 门控

RGB 是欧氏绝对坐标，三通道具有不可拆分的从属关系。任意单通道达标都不能单独写入颜色原型，必须整体有效或整体丢弃。

二维平滑度推荐由局部颜色方差或局部颜色梯度定义：
$$
V_{rgb}(\mathbf{p})
=
\frac{1}{|\mathcal{N}|}
\sum_{\mathbf{q}\in\mathcal{N}(\mathbf{p})}
\left\|
\mathbf{x}_{rgb}(\mathbf{q})
-
\bar{\mathbf{x}}_{rgb}(\mathbf{p})
\right\|^2
$$
$$
G^{rgb}(\mathbf{p})
=
\left[V_{rgb}(\mathbf{p})<\tau_{rgb\_var}\right]
\land
\left[g(\mathbf{p})<\tau_{edge\_block}\right]
$$

其中第二项用于避免正落在强边界上时把混合像素写成新的面域颜色。若系统需要学习纹理而非纯面域，可保留颜色写入，但应把 topology 标记为 `Texture_Surface`，不能与 `Surface_2D` 原型混用。

#### 3.3 `Curv` 门控

曲率只有在存在稳定边界或一维 trace 支持时才有物理意义。对每个尺度 $s$：
$$
G^{curv}_s(\mathbf{p})
=
G^{parent}_{trace/boundary,s}(\mathbf{p})
\land
\left[|curv_s(\mathbf{p})|>\tau_{curv,s}\right]
\land
\left[conf_{curv,s}(\mathbf{p})>\tau_{curv\_conf,s}\right]
$$

各尺度曲率可以独立有效，但必须受父边界/线域门控约束。若父门控失败，不允许单独写入曲率节点；否则会把噪声纹理中的局部二阶响应误注册为轮廓事件。

#### 3.4 `aps` 门控

`aps` 表示局部形状比例或尺度化长宽结构，不应在没有足够空间支持的单点上写入。对尺度 $s$：
$$
G^{aps}_s(\mathbf{p})
=
\left[A_s(\mathbf{p})>\tau_{area,s}\right]
\land
\left[occ_s(\mathbf{p})>\tau_{occ,s}\right]
\land
\left[conf_{aps,s}(\mathbf{p})>\tau_{aps\_conf,s}\right]
$$

其中 $A_s$ 是对应尺度下的连通域面积或有效采样面积，$occ_s$ 是矢量占有度/区域占有率。`aps` 的相似度可独立按尺度降维计算，但至少需要满足 `min_valid_aps_scales`，否则不参与匹配。

#### 3.5 `ori` 门控

`ori` 是显式周期属性，适合描述稳定线域或边界轴向。对尺度 $s$：
$$
R_{ori,s}(\mathbf{p})
=
\frac{
\left|
\sum_{\mathbf{q}\in\mathcal{N}_s(\mathbf{p})}
w_s(\mathbf{q})\exp(i k_{ori}\theta_s(\mathbf{q}))
\right|
}{
\sum_{\mathbf{q}\in\mathcal{N}_s(\mathbf{p})}w_s(\mathbf{q})+\epsilon
}
$$
$$
G^{ori}_s(\mathbf{p})
=
G^{parent}_{trace/boundary,s}(\mathbf{p})
\land
\left[R_{ori,s}(\mathbf{p})>\tau_{ori\_coh,s}\right]
$$

`ori` 默认使用 $k_{ori}=2$，表示 $\theta$ 与 $\theta+\pi$ 为同一轴向；若某任务确实需要方向箭头而非轴向，再切换为 $k_{ori}=1$。

### 4. 分组相似度检索公式

相似度不再由 `mod_id` 直接选择单个算子，而由模态内的相似度组计算后合成。设模态 $m$ 的有效相似度组为 $\mathcal{G}_m$，则：
$$
S_m
=
\exp
\left(
\frac{
\sum_{g\in\mathcal{G}_m}\alpha_g\log(S_g+\epsilon)
}{
\sum_{g\in\mathcal{G}_m}\alpha_g+\epsilon
}
\right)
$$

使用加权几何平均而不是简单求和，是为了避免某个容易匹配的通道掩盖另一个物理上失败的通道。

#### 4.1 欧氏绝对坐标

用于 `RGB`、`Curv`、`aps`：
$$
S_E
=
\exp
\left(
-
\frac{
\sum_c m_c w_c
\left(\frac{x_c-\mu_c}{\sigma_c}\right)^2
}{
2\left(\sum_c m_c w_c+\epsilon\right)
}
\right)
$$

`RGB` 应使用整体三通道 RBF；`Curv` 与 `aps` 可以按有效尺度掩码降维，但每个尺度必须有独立 $\sigma_c$ 和权重 $w_c$。

#### 4.2 矢量占有度 / 梯度方向

用于 `grad.c2,c3`。先由相似度通道构造向量：
$$
\mathbf{v}=(g_x,g_y),\quad
\hat{\mathbf{v}}=\frac{\mathbf{v}}{\|\mathbf{v}\|+\epsilon}
$$

若使用极性敏感匹配：
$$
S_V
=
\max(0,\hat{\mathbf{v}}_x\cdot\hat{\mathbf{v}}_w)^{\gamma_{grad}}
$$

若使用轮廓轴向匹配，推荐：
$$
S_V
=
|\hat{\mathbf{v}}_x\cdot\hat{\mathbf{v}}_w|^{\gamma_{grad}}
$$

`grad.c0` 的强度不直接进入 $S_V$，但可作为节点更新权重或 Gpos 响应门控：
$$
S^{Gpos}_{grad}(\mathbf{q})
=
G^{grad}(\mathbf{q})\cdot S_V(\mathbf{q})
$$

这能避免 `atan2` 角度被点积误用，也能避免不同对比度的同向边缘被错误拆成大量节点。

#### 4.3 周期属性

用于 `ori`：
$$
S_P
=
\left(
\frac{1+\cos(k(\theta_x-\theta_w))}{2}
\right)^{\gamma_{ori}}
$$

其中 $k=2$ 表示轴向周期为 $\pi$，$k=1$ 表示方向周期为 $2\pi$。每个尺度的 `ori` 应配置独立 $\gamma_{ori,s}$、$\theta_{match,ori,s}$ 与最小 coherence。

### 5. 推荐配置结构

代码层面建议将当前 `feature_types` 与 `feature_attributes` 升级为显式 schema，而不是仅用 `STRENGTH / CONTINUITY_SURFACE / CONTINUITY_TRACE` 三类字符串推断行为。示例：

```python
feature_specs = {
    0: {
        "name": "grad",
        "topology": "Boundary_Edge",
        "channels": {
            0: {"name": "abs_strength", "role": "gate_only", "gate": "grad_strength"},
            1: {"name": "theta", "role": "context_only", "gate": "grad_coherence"},
            2: {"name": "gradx", "role": "sim_only", "metric_group": "grad_vec", "parents": ["grad_strength", "grad_coherence"]},
            3: {"name": "grady", "role": "sim_only", "metric_group": "grad_vec", "parents": ["grad_strength", "grad_coherence"]},
        },
        "metric_groups": {
            "grad_vec": {"metric": "Vector_Occupancy", "channels": [2, 3], "gamma": 2.0, "polarity": "axis", "theta_match": 0.75}
        },
    },
    1: {
        "name": "RGB",
        "topology": "Surface_2D",
        "channels": {
            0: {"name": "R", "role": "gate_and_sim", "metric_group": "rgb"},
            1: {"name": "G", "role": "gate_and_sim", "metric_group": "rgb"},
            2: {"name": "B", "role": "gate_and_sim", "metric_group": "rgb"},
        },
        "gates": {"surface": {"metric": "local_variance", "tau": "tau_rgb_var"}},
        "metric_groups": {
            "rgb": {"metric": "Euclidean_Absolute", "channels": [0, 1, 2], "sigma": [0.06, 0.06, 0.06], "theta_match": 0.85}
        },
    },
}
```

该 schema 的关键要求是：`gate`、`context` 与 `metric_group` 显式分离。实现中可以继续保留 `feature_types` / `feature_attributes` 作为兼容层，但最终应由 schema 生成门控掩码、相似度掩码和 Gpos 响应计算，而不是在 `_extract_and_match_local_features()` 中硬编码三类通道属性。

### 6. 写入与投影工作流修订

规范化后的 `_extract_and_match_local_features()` 应按以下顺序执行：

1.  `evaluate_gates(mod_id, X_m, p)`：计算父门控和子通道门控。
2.  `build_descriptor(mod_id, X_m, p, gates)`：生成 `sim_values`、`sim_mask`、`gate_context`、`topology_context`。
3.  `match_descriptor(descriptor, nodes)`：按 `metric_groups` 计算分组相似度，使用模态级或角色级 `theta_match`。
4.  `register_or_update()`：命中则更新原型统计，未命中则新增 GmemI 节点；无论命中与否，若处于 MICRO 对象学习中，都应写入 GmemII observation relation。
5.  `project_gpos()`：GposI 响应使用同一 descriptor schema，形式为：
$$
S^{I}_{node}(\mathbf{q})
=
G_m(\mathbf{q})
\cdot
S_m(\mathbf{q};\mathbf{W}_{node})
$$

其中 $G_m$ 是该模态的门控场。对 `grad` 而言，$G_m=G^{grad}$；对 `RGB` 而言，$G_m=G^{rgb}$。这样，门控通道虽然不参与相似度，却仍能限制响应只出现在物理有效区域。

### 7. 最小落地顺序

若工程组希望以最小改动修复当前 `grad` 检索失效，建议按以下顺序实现：

1.  将 `grad` 输入扩展为 `[abs_strength, theta, gradx, grady]`。
2.  把 `grad` 的相似度 mask 固定为只包含 `[gradx, grady]`，`abs_strength` 与 `theta` 只进入门控和上下文。
3.  将 `Vector_Magnitude` 点积替换为归一化向量相似度，并默认使用轴向匹配 $|\hat{\mathbf{v}}_x\cdot\hat{\mathbf{v}}_w|^{\gamma}$。
4.  为 `RGB`、`Curv`、`aps`、`ori` 设置独立 `theta_match`、`sigma`、`gamma`、`min_valid_channels`，取消全局固定 `similarity_threshold=0.85`。
5.  在 GposI 投影中乘回门控场，保证只在物理有效区域产生响应。

完成以上步骤后，`grad` 不会再因 `atan2` 角度被点积误用而失效；同时，颜色、曲率、长宽比和朝向也会遵循同一套“先门控、后分组相似度、再写入/投影”的规范。

---

## 补充：过渡带问题的统一图表征修订

### 1. 设计约束

本节覆盖上一版“渐变边界、语义前沿蚕食与意图扩展”方案。上一版方案能够解释调试现象，但引入了 `FRONTIER_ADVANCE`、`TRANSITION_PROBE`、`BAND_TRACE` 等专用意图，并将过渡带写成 GmemII 中的特殊关系载荷。这与本项目的核心设想不够一致。

更符合当前项目初衷的约束应为：

1.  **节点存储感官刺激**：视觉中即 CNN 或固定算子呈递的局部特征响应。
2.  **边存储关系**：主要是相对位置、时间顺序、共现频率与置信度，不把复杂视觉剖面塞进边中。
3.  **高层由低层组合而来**：过渡带、轮廓、面域和对象都应是低层特征节点及其时空关系的组合结果。
4.  **眼动由响应场、疲劳、缺口与关系预期涌现**：尽量避免为每种视觉现象增加一个手写意图。

因此，渐变边界不应被建模为特殊的 `transition_band` 关系边，也不应主要依赖 semantic 蚕食状态机。它应被建模为一种低层视觉刺激：**有宽度、有方向、有颜色变化剖面的连续过渡特征**。该刺激由定参数 CNN/固定滤波器产生，进入 GmemI 成为普通特征节点；GmemII 只负责记录它与 surface、edge、corner 等节点之间的相对位置和时间组合。

### 2. 对三类思路的重新评判

#### 2.1 一维连续与二维连续统一蚕食

“统一蚕食”的优点是看到了 surface 与 edge 都是连续响应场，而不是完全不同的对象。但如果把所有拓扑都统一成 semantic 区域扩张，会丢失图结构的颗粒度：线、面、点事件在视觉刺激上本来就是不同响应族，不应由一个区域算法机械解释。

更合理的统一方式不是统一成一个 semantic mask，而是统一成一个**特征响应流形 (feature response manifold)**：

$$
\mathcal{M}_i=\{\mathbf{q}\mid S_i(\mathbf{q})>\tau_i\}
$$

其中 $S_i$ 是某个 GmemI 节点在 GposI 中的响应。RGB surface、局部硬边、曲率点、方向线段、渐变过渡都只是不同节点的响应流形；差别体现在节点的 `support_dim`、`support_scale`、`metric_group` 和门控函数上，而不是体现在不同的专用状态机上。

结论：保留“连续响应逐步覆盖”的思想，但取消“所有东西都用 semantic 蚕食解释”的倾向。统一对象应是 GmemI/GposI 响应流形，而不是二值 semantic。

#### 2.2 增加多尺度信息

用户担心固定特征金字塔会破坏连续性，这个担心成立：如果每个尺度都被当作独立模态或独立图层，会制造人为离散层级和跨尺度合并问题。

但增加定参数滤波器本身是合理的。人类早期视觉也不是只看单尺度梯度，而是存在不同感受野大小、方向和频率选择性的神经元。关键区别是：尺度不应成为新的离散语义层，而应成为特征节点中的一个连续坐标。

推荐使用**连续尺度近似的固定滤波族**：

$$
K_{\theta,\sigma}^{ramp},\quad
\theta\in[0,\pi),\quad
\sigma\in[\sigma_{min},\sigma_{max}]
$$

工程上可以用少量离散 $\sigma$ 采样，但写入节点时保存的是经邻近尺度插值得到的 $\log\sigma^*$，而不是保存“第 1 层/第 2 层/第 3 层金字塔”。这样既允许宽渐变被大感受野捕获，又避免把连续变化硬切成多个特征层级。

结论：不采用笨重的特征金字塔；采用固定、可解释、尺度坐标连续化的过渡特征滤波族。

#### 2.3 动态阈值

动态阈值是必要的，但它不应成为解决问题的主体。固定阈值会漏掉低对比渐变；简单降低阈值又会吞入噪声和 padding 黑边。更干净的做法是把阈值改成所有底层特征统一使用的**神经归一化增益**：

$$
Z_i(\mathbf{q})
=
\frac{
E_i(\mathbf{q})-\operatorname{median}_{\mathbf{u}\in V}E_i(\mathbf{u})
}{
\operatorname{MAD}_{\mathbf{u}\in V}E_i(\mathbf{u})+\epsilon
}
$$

其中 $E_i$ 是某个固定滤波器的原始能量，$V$ 是有效视野区域。门控使用 $Z_i>\tau_i^z$，而不是直接使用原始强度。这样，弱但稳定的渐变可在低噪声区域被激活；高噪声纹理则不会因为绝对强度略高而大量写入。

结论：动态阈值作为统一的底层归一化机制存在，不作为 `BOUNDARY_SEEK` 或某个意图的特殊逻辑。

### 3. 新增低层特征族：`RGB_TRANSITION`

为解决 $A(\text{white})\rightarrow B(\text{gray})\rightarrow C(\text{black})$ 这类宽过渡问题，新增一个普通低层特征族：`RGB_TRANSITION`。它不是 GmemII 特殊边，也不是 MICRO 特殊状态，而是和 `RGB`、`grad`、`curv`、`ori` 同级的视觉刺激节点来源。

#### 3.1 固定滤波器定义

对每个方向 $\theta$ 和尺度 $\sigma$，定义法向 $\hat{\mathbf{n}}_\theta$ 与切向 $\hat{\mathbf{t}}_\theta$。`RGB_TRANSITION` 使用沿切向平滑、沿法向比较两侧颜色分布的双叶或斜坡滤波器：

$$
\mathbf{L}_{+}(\mathbf{p};\theta,\sigma)
=
\int
\mathbf{x}_{rgb}(\mathbf{p}+u\hat{\mathbf{t}}_\theta+s\hat{\mathbf{n}}_\theta)
G_{\sigma_t}(u)
G_{\sigma}(s-d_\sigma)
\,du\,ds
$$

$$
\mathbf{L}_{-}(\mathbf{p};\theta,\sigma)
=
\int
\mathbf{x}_{rgb}(\mathbf{p}+u\hat{\mathbf{t}}_\theta+s\hat{\mathbf{n}}_\theta)
G_{\sigma_t}(u)
G_{\sigma}(s+d_\sigma)
\,du\,ds
$$

颜色过渡向量：

$$
\Delta_{\theta,\sigma}(\mathbf{p})
=
\mathbf{L}_{+}(\mathbf{p};\theta,\sigma)
-
\mathbf{L}_{-}(\mathbf{p};\theta,\sigma)
$$

能量：

$$
E_{tr}(\mathbf{p};\theta,\sigma)
=
\|\Delta_{\theta,\sigma}(\mathbf{p})\|_2
$$

该滤波器对单像素硬边有响应，对宽渐变也有响应；差别只体现在最优尺度 $\sigma^*$ 和过渡宽度上。

#### 3.2 单调性与连续性门控

为了避免纹理噪声被误认为渐变过渡，`RGB_TRANSITION` 需要一个沿法向的单调性门控。沿 $\hat{\mathbf{n}}_\theta$ 采样 $K$ 个点，设相邻颜色差为：

$$
\delta_k=\mathbf{x}_{rgb}(\mathbf{p}+s_{k+1}\hat{\mathbf{n}}_\theta)
-
\mathbf{x}_{rgb}(\mathbf{p}+s_k\hat{\mathbf{n}}_\theta)
$$

以总过渡方向 $\hat{\Delta}=\Delta/\|\Delta\|$ 为参考，定义：

$$
C_{mono}
=
\frac{
\sum_k \max(0,\langle \delta_k,\hat{\Delta}\rangle)
}{
\sum_k |\langle \delta_k,\hat{\Delta}\rangle|+\epsilon
}
$$

若 $C_{mono}$ 高，说明颜色沿法向主要朝同一方向变化；若低，则更可能是纹理、噪声或重复图案。

最终门控：

$$
G^{tr}(\mathbf{p})
=
\left[
Z_{tr}(\mathbf{p};\theta^*,\sigma^*)>\tau_{tr}^z
\right]
\land
\left[
C_{mono}(\mathbf{p};\theta^*,\sigma^*)>\tau_{mono}
\right]
\land
M_{valid}(\mathbf{p})
$$

其中：

$$
(\theta^*,\sigma^*)
=
\arg\max_{\theta,\sigma}
Z_{tr}(\mathbf{p};\theta,\sigma)\cdot C_{mono}(\mathbf{p};\theta,\sigma)
$$

### 4. `RGB_TRANSITION` 节点描述符

`RGB_TRANSITION` 写入 GmemI 时应遵守探讨二修订版中的通道契约。推荐描述符：

| 通道组 | 内容 | role | metric |
| --- | --- | --- | --- |
| `energy` | $E_{tr}$ 或 $Z_{tr}$ | `gate_only` | `None` |
| `mono` | $C_{mono}$ | `gate_only` | `None` |
| `contrast_vec` | $\Delta/\|\Delta\|$ 或未归一化 $\Delta$ | `sim_only` | `Vector_Occupancy` / `Euclidean_Absolute` |
| `width` | $\log\sigma^*$ | `sim_only` | `Euclidean_Absolute` |
| `axis` | $\theta^*$ 或 $(\cos 2\theta^*,\sin 2\theta^*)$ | `sim_only` | `Periodic_Property` |

相似度采用分组几何平均：

$$
S_{tr}
=
S_{\Delta}^{\alpha_\Delta}
\cdot
S_{\sigma}^{\alpha_\sigma}
\cdot
S_{\theta}^{\alpha_\theta}
$$

其中：

$$
S_{\Delta}
=
\left|\hat{\Delta}_x\cdot\hat{\Delta}_w\right|^{\gamma_\Delta}
$$

$$
S_{\sigma}
=
\exp\left(
-\frac{(\log\sigma_x-\log\sigma_w)^2}{2\sigma_{scale}^2}
\right)
$$

$$
S_{\theta}
=
\left(
\frac{1+\cos(2(\theta_x-\theta_w))}{2}
\right)^{\gamma_\theta}
$$

注意：过渡带的颜色剖面属于节点特征，不属于边。边只记录该过渡刺激与其他刺激之间的位置和时间关系。

### 5. 图结构中的表达方式

以 $A(\text{white})\rightarrow B(\text{gray})\rightarrow C(\text{black})$ 为例，理想写入不是：

$$
e_{band}=(n_A,n_C,\text{band profile})
$$

而是：

$$
n_A^{rgb},\quad
n_{A\rightarrow C}^{tr},\quad
n_B^{rgb?},\quad
n_C^{rgb?}
$$

以及它们之间的普通时空关系边：

$$
e_1=(n_A^{rgb}, n_{A\rightarrow C}^{tr}, \Delta\mathbf{p}_1, \Delta t_1)
$$

$$
e_2=(n_{A\rightarrow C}^{tr}, n_C^{rgb}, \Delta\mathbf{p}_2, \Delta t_2)
$$

若灰色 $B$ 区域足够宽且稳定，会自然生成 $n_B^{rgb}$；若 $B$ 只是窄过渡中的中间值，则不会被强行写成 surface 节点，而会保留在 `RGB_TRANSITION` 节点的局部剖面中。这样高层图仍能追溯到低层刺激：白色面、颜色过渡、黑色面分别是可再现的节点；它们的相邻触发则由边完成联想。

对于复杂形状，多个 `RGB_TRANSITION` 观察会复用同类过渡原型节点，或在方向、宽度、颜色变化不同处形成不同原型节点。GmemII 中的高层组合只是这些节点实例沿空间和时间形成的子图，不需要额外的 `transition_band` 专用结构。

### 6. 眼动引导的统一化

不再为渐变边界新增专用意图。MICRO 可以被简化为同一个局部图补全过程：当前活跃图结构中，已解释的响应被疲劳抑制，未解释但稳定的特征响应获得眼跳增益。

对所有特征族 $m$，令 $R_m(\mathbf{q})$ 为底层门控后的原始响应，令 $\hat{R}_m(\mathbf{q})$ 为当前已存在 GmemI/GposI 节点能够解释的响应：

$$
\hat{R}_m(\mathbf{q})
=
\max_{n_i\in\mathcal{A}_m} S_i^{Gpos}(\mathbf{q})
$$

新奇残差：

$$
N_m(\mathbf{q})
=
R_m(\mathbf{q})\cdot(1-\hat{R}_m(\mathbf{q}))
$$

通用 MICRO drive：

$$
U_{micro}(\mathbf{q})
=
\sum_m w_m
\left[
\lambda_s S_m^{known}(\mathbf{q})
+
\lambda_n N_m(\mathbf{q})
+
\lambda_g G_m^{relation\_gap}(\mathbf{q})
\right]
\odot
(1-O_{visited}(\mathbf{q}))
\odot
M_{valid}(\mathbf{q})
$$

其中：

*   $S_m^{known}$ 使系统能沿已知响应流形继续采样，保证再现和稳定化。
*   $N_m$ 使系统看向当前图还不能解释的新刺激。
*   $G_m^{relation\_gap}$ 来自 GmemII 中尚未闭合或置信度低的关系边，负责联想式补全。
*   $M_{valid}$ 是感官有效区域，不是对象语义补丁；padding 区域不产生可学习刺激节点。

`Surface_2D`、`Boundary_Edge`、`Trace_1D`、`RGB_TRANSITION` 的差异只通过 `support_dim` 与空间约束函数体现：

| support_dim | 典型特征 | 空间约束 |
| --- | --- | --- |
| 2 | RGB surface、aps 区域 | 蓝噪声覆盖、低重复采样 |
| 1 | grad edge、ori line、RGB transition | 切向延续、端点/曲率优先 |
| 0 | corner、junction、terminator | 稀疏事件吸引 |

这使“看向边缘”“沿边界走”“离开重复面域”不再是手写意图，而是不同支持维度的响应流形在同一 drive 公式中的自然结果。

### 7. 外围黑边问题的结构化处理

外围黑边不应通过 `BOUNDARY_SEEK` 特殊规则排除，而应在输入层作为**无效感官区域**处理。若图像经过 padding 到 $H\times W$，预处理必须同步输出：

$$
M_{valid}(\mathbf{q})\in\{0,1\}
$$

所有固定滤波器、GmemI 写入、GposI 投影和 MICRO drive 都必须乘以 $M_{valid}$。这意味着 padding 黑边不会产生 RGB 节点、grad 节点或 `RGB_TRANSITION` 节点，也不会成为可学习关系的一端。

若暂时无法从预处理获得 `M_{valid}`，可将与画幅边缘连通、颜色近似常量、形状近似矩形且面积超过阈值的区域标为 `invalid_candidate`。但这只是输入有效性恢复，不应写入 GmemII，也不应作为对象边界事件。

### 8. 落地顺序

推荐以结构一致性为优先，而不是继续增加意图分支：

1.  在特征提取层新增 `RGB_TRANSITION` 固定滤波族，使用方向 $\theta$ 与连续化尺度 $\log\sigma$ 表征宽渐变。
2.  将 `RGB_TRANSITION` 纳入现有通道契约：门控通道只负责能量和单调性，相似度通道负责颜色变化向量、宽度和轴向。
3.  为 GmemI 节点增加或明确使用 `support_dim`、`support_scale`、`metric_groups`，让 surface、edge、transition、corner 的差异体现为节点属性。
4.  修改 GposI 投影，使 `RGB_TRANSITION` 和其他特征一样产生响应图；眼动只看统一的响应、残差、疲劳和关系缺口。
5.  逐步把 `BOUNDARY_SEEK`、`CONTOUR_TRACE` 等显式意图降级为调参模式或调试标签，最终由通用 MICRO drive 统一选择下一注视点。
6.  在预处理或特征输入层加入 `M_{valid}`，禁止 padding 区域产生节点和 drive。

该方案的核心不是为过渡带增加特殊高层语义，而是补齐底层视觉特征族：既有对点、线、面敏感的节点，也有对宽颜色变化剖面敏感的节点。高层图仍只由节点集合和时空关系边构成；再现依赖节点回放，联想依赖边触发，过渡带只是其中一种可被再现和联想的低层刺激。

### 9. 对通用 MICRO drive 的评判与增强方案

第 6 小节提出的通用 MICRO drive 与本项目的核心思想是相容的：节点保存感官刺激，边保存时空关系，眼动不依赖人为划分的专用任务，而由当前响应、残差、疲劳和关系缺口共同涌现。它比 `SURFACE_CONFIRM`、`BOUNDARY_SEEK`、`CONTOUR_TRACE` 等显式意图更接近统一结构，也更适合后续向“再现”与“联想”扩展。

但第 6 小节若直接激进落地，会带来三类风险：

1.  **支持维度竞争失衡**：二维 surface 的响应面积大，若与一维 edge/transition 直接求和，argmax 容易长期落在面域内部；反过来，强硬边响应过高时，也可能压制内部颜色统计。
2.  **缺少关系持续性**：边缘追踪并不只是“下一点哪里边缘强”，还依赖上一眼到这一眼形成的临时关系方向。若没有“正在形成的关系边”作为短时约束，统一 drive 会在局部高响应点间跳跃，导致追踪断裂。
3.  **新奇残差易吸噪声**：$N_m=R_m(1-\hat{R}_m)$ 能发现未知刺激，但若没有持续性、连通性和有效视野约束，纹理噪声、padding 伪边缘或孤立异常点会被误认为新节点。

因此，推荐的折中方案是：**保留一个统一 MICRO drive，但将一维、二维、点状差异降级为节点属性和关系边参数，而不是手写意图状态。** 换言之，系统不再问“现在处于哪个意图”，而是问“当前活跃节点和未闭合关系边各自产生怎样的空间约束核”。

#### 9.1 统一 drive 的颗粒度修正

对每个 GmemI 节点 $n_i$，增加或明确以下低层描述字段：

$$
\chi_i =
\left(
support\_dim_i,\,
support\_axis_i,\,
support\_scale_i,\,
metric\_group_i,\,
confidence_i
\right)
$$

其中 $support\_dim_i$ 可以取连续近似值或离散近似值：$0$ 表示角点/端点等稀疏事件，$1$ 表示边缘/线段/过渡带，$2$ 表示面域。该字段不是语义类别，而是该刺激响应流形的几何维度。

统一 drive 应改写为“节点响应项 + 新奇残差项 + 关系缺口项 + 拓扑空间核”的乘性组合：

$$
U_{micro}(\mathbf{q})
=
M_{valid}(\mathbf{q})
\odot
\left(1-O_{loc}(\mathbf{q})\right)
\odot
\sum_i
\eta_i(t)
\left[
\lambda_s S_i^{known}(\mathbf{q})
+
\lambda_n N_i(\mathbf{q})
+
\lambda_g G_i^{relation\_gap}(\mathbf{q})
\right]
\odot
K_i^{topo}(\mathbf{q}\mid \chi_i,\mathcal{H}_t)
$$

其中 $\mathcal{H}_t$ 是短时观察历史，$K_i^{topo}$ 是由节点支持维度、轴向、尺度和正在形成的关系边共同确定的空间约束核。这样一来，“看面域内部”“跳向边界”“沿边界走”不是不同意图，而是不同 $\chi_i$ 和关系边在同一个公式中产生的不同核函数。

为了避免二维和一维响应直接竞争导致能量偏置，应先按支持维度归一化，再汇合：

$$
U_d(\mathbf{q})
=
\operatorname{Norm}
\left(
\sum_{i:support\_dim_i=d}
\eta_i(t)
F_i(\mathbf{q})
\odot
K_i^{topo}(\mathbf{q})
\right)
$$

$$
U_{micro}(\mathbf{q})
=
M_{valid}(\mathbf{q})
\odot
\left(1-O_{loc}(\mathbf{q})\right)
\odot
\sum_{d\in\{0,1,2\}}
\alpha_d(t)U_d(\mathbf{q})
$$

其中 $\alpha_d(t)$ 不应是固定意图标签，而应由未解释残差、关系缺口和覆盖度自动决定：

$$
\alpha_d(t)
=
\operatorname{Softmax}_d
\left(
b_d
+ \kappa H_d^{res}
+ \mu H_d^{gap}
- \nu C_d^{covered}
\right)
$$

这样可以保证 surface、edge、transition、corner 在统一 drive 中公平竞争，同时保留对当前最缺信息维度的偏置。

#### 9.2 拓扑空间核的统一定义

不同支持维度只改变 $K_i^{topo}$ 的参数，不改变控制器结构。

**二维面域核**：

$$
K_{2D}(\mathbf{q})
=
M_i(\mathbf{q})
\odot
D_{blue}(\mathbf{q},\mathcal{P}_i)
\odot
\left(1-O_i^{wide}(\mathbf{q})\right)
$$

它鼓励在同一响应流形中做少量代表性采样，用于学习颜色均值、方差和内部残差，不负责逐点扫完整个面域。

**一维线域/过渡带核**：

$$
K_{1D}(\mathbf{q})
=
S_i^{Gpos}(\mathbf{q})
\odot
\exp\left(
-\frac{\angle(\mathbf{q}-\mathbf{p}_t,\hat{\mathbf{t}}_e)^2}{2\sigma_\theta^2}
\right)
\odot
\exp\left(
-\frac{(\|\mathbf{q}-\mathbf{p}_t\|-s_e)^2}{2\sigma_s^2}
\right)
\odot
\left(1-O_e^{thin}(\mathbf{q})\right)
$$

其中 $\hat{\mathbf{t}}_e$ 与 $s_e$ 来自正在形成的关系边，而不是来自 `CONTOUR_TRACE` 这种手写状态。该核保证边缘追踪具有方向惯性、步长约束和局部连续性。

**点事件核**：

$$
K_{0D}(\mathbf{q})
=
S_i^{Gpos}(\mathbf{q})
\odot
\left(1-O_i^{point}(\mathbf{q})\right)
$$

它用于角点、端点、交点等稀疏事件，抑制半径应小而强，避免点事件被反复注视。

#### 9.3 用“正在形成的关系边”替代显式追踪意图

边缘追踪失败的主要原因通常不是缺少 `CONTOUR_TRACE` 标签，而是缺少短时关系惯性。推荐在 MICRO 过程中维护一个临时关系边：

$$
e_t^{work}
=
\left(
src,\,
dst,\,
\Delta\mathbf{p}_t,\,
\hat{\mathbf{t}}_t,\,
support\_dim,\,
age,\,
confidence
\right)
$$

当连续两次注视命中同一一维响应流形，或命中相容的 `grad/ori/RGB_TRANSITION` 节点时，更新该工作边：

$$
\hat{\mathbf{t}}_{t+1}
=
\operatorname{Norm}
\left(
(1-\beta)\hat{\mathbf{t}}_t
+ \beta
\frac{\mathbf{p}_{t+1}-\mathbf{p}_t}
{\|\mathbf{p}_{t+1}-\mathbf{p}_t\|+\epsilon}
\right)
$$

$$
confidence_{t+1}
=
\rho confidence_t
+ (1-\rho)
S_i^{Gpos}(\mathbf{p}_{t+1})
C_{align}(\mathbf{p}_{t+1})
$$

该工作边未必立即写入长期 GmemII；只有当 $age$、$confidence$ 和空间跨度达到阈值后，才固化为正式关系边或自指边。这样，边缘追踪是由“关系边正在闭合”驱动的，而不是由 `BOUNDARY_SEEK -> CONTOUR_TRACE` 的状态切换驱动的。

进入和退出也应使用当前注视点证据，而不是使用全局 `drive.max()`：

$$
H_i(t)
=
S_i^{Gpos}(\mathbf{p}_t)
\cdot
C_{align}(\mathbf{p}_t)
\cdot
M_{valid}(\mathbf{p}_t)
$$

若 $H_i(t)>\theta_{enter}$，建立或增强工作边；若连续 $L$ 步 $H_i(t)<\theta_{exit}$，才衰减或关闭该工作边，且 $\theta_{enter}>\theta_{exit}$。这提供迟滞，避免边缘追踪一两步后早退。

#### 9.4 关系缺口项的具体化

第 6 小节中的 $G_m^{relation\_gap}$ 是正确方向，但需要更明确地绑定图结构。对未闭合关系边 $e=(src\rightarrow dst)$，其缺口场可定义为：

$$
G_e^{gap}(\mathbf{q})
=
(1-confidence_e)
\cdot
S_{dst}^{Gpos}(\mathbf{q})
\cdot
\exp\left(
-\frac{1}{2}
d_e(\mathbf{q})^T\Sigma_e^{-1}d_e(\mathbf{q})
\right)
$$

其中 $d_e(\mathbf{q})$ 是候选点与该关系边期望相对位置的差异。若 $dst$ 尚未确定，则用相容支持维度的响应族替代：

$$
S_{dst}^{Gpos}(\mathbf{q})
\leftarrow
\sum_{j:support\_dim_j\sim support\_dim_e}
\omega_j S_j^{Gpos}(\mathbf{q})
$$

这使联想不只是高层节点激活，而是具体表现为“未完成关系边”对下一眼的牵引：缺什么关系，就看向最可能补全这个关系的位置。

#### 9.5 疲劳与访问抑制的分化

统一 drive 不应只使用一个 $O_{visited}$。否则二维面域的大范围抑制会破坏一维轮廓追踪，或者一维追踪的局部抑制无法阻止面域内部重复采样。推荐分成三类：

$$
O_{visited}
=
w_l O_{loc}
+ w_n O_{node}
+ w_e O_{edge}
$$

*   $O_{loc}$：位置级短时返回抑制，所有支持维度共享。
*   $O_{node}$：节点级疲劳，抑制同一 GmemI 原型的重复注视。
*   $O_{edge}$：关系边级疲劳，只抑制已经闭合或高置信的关系片段。

二维节点应使用较宽的 $O_{node}^{wide}$ 以快速减少内部重复；一维关系边应使用较窄且沿已走过轨迹分布的 $O_{edge}^{thin}$，避免把前方连续边缘一并压掉。

#### 9.6 推荐落地方式

为了避免一次性替换导致追踪崩溃，落地应分三步：

1.  **保留现有 `MicroIntention` 作为调试标签**，但逐步把它从状态机分支降级为 drive 权重观测值。先让现有实现输出各支持维度的 $U_0,U_1,U_2$ 和 $\alpha_d$，验证统一 drive 是否选择同样或更合理的落点。
2.  **引入工作关系边**，用其提供 $K_{1D}$ 的切向、步长和迟滞。只有这一点稳定后，才减少 `BOUNDARY_SEEK/CONTOUR_TRACE` 的硬切换逻辑。
3.  **按支持维度归一化与分化疲劳**，防止 surface 面积优势、edge 强响应优势或噪声残差主导整个 drive。

最终目标不是取消一维与二维的差异，而是把差异放回正确颗粒度：它们是不同低层刺激节点的响应流形和支持维度，不是两套手写眼动程序。这样既保留第 6 小节的统一性，也能通过关系边惯性、迟滞和支持维度归一化降低边缘追踪失败的风险。

---

## 补充：新优化器 REVIEW 周期抢占与重复眼跳的审查（2026-09-09）

本节审查 `src/nns/memorygraphs/graph_memorypool_newoptimizer.py` 的当前实现，并结合 `src/test_memorypoolForJune2.ipynb` 已保存的输出讨论修正方向。以下算法修改尚未实施；参数建议是定位实验，不代表已验证的最优配置。源码位置按本次审查时的行号记录。

### 1. 现象与证据边界

Notebook 的 50 步保存输出中，运行落点共有 7 个不同坐标，不计手动设置的初始点。从第 16 步开始仅剩 `(389, 195)` 与 `(373, 241)` 两点往返。第 4 步以后，所有打印为 `LEARN_MICRO` 或 `LEARN_MACRO` 的步骤均没有相对上一步移动。

最后保存的状态竞争输入与得分如下：

| 观测项 | 数值 |
| --- | --- |
| 窗口外召回 `recall_out` / 窗口内召回 `recall_in` | 0.997843 / 1.0 |
| 新颖度 / 前沿 / 有效视域占比 | 0.124667 / 0.343274 / 1.0 |
| 失败量 `fail` / 完成度 / 恢复价值 | 1.0 / 0.0 / 0.0 |
| MACRO / MICRO / REVIEW 得分 | 1.046460 / 0.349062 / 1.896981 |

这些日志只记录步骤结束后的兼容状态 `controller.state`，没有记录每步实际执行的处理方法。因此不能把奇数步的 `LEARN_*` 直接解释为真正执行了一步学习。

使用源码中的 `_score_optimizer_states()`、`_select_behavior_state()`、`_set_behavior_state()`、`_run_review_verify()` 和 `_bfs_recognize()`，以保存的观测量替代图像场、以一个共享特征构造简化语义，已隔离复现以下循环。该验证不依赖完整图像前端，也不等同于重新运行原图的全部轨迹。

```text
步骤开始状态       实际执行方法          步骤结束状态       位置变化
MACRO_SEARCH       _run_review_verify    REVIEW_VERIFY      A -> B
REVIEW_VERIFY      _run_review_verify    MACRO_SEARCH       停在 B
MACRO_SEARCH       _run_review_verify    REVIEW_VERIFY      B -> A
REVIEW_VERIFY      _run_review_verify    MACRO_SEARCH       停在 A
```

### 2. 主要缺陷及其因果关系

#### 2.1 REVIEW 抢占绕过滞回，后继状态尚未执行就再次被抢占

位置：`_select_behavior_state()` L1922、`_run_review_verify()` L3868、`_run_suspend_resume()` L3914、`run_step()` L3975。

REVIEW 抢占判断在 `state_min_age` 和 `state_switch_margin` 检查之前。REVIEW 成功后，处理方法直接将状态改为 MACRO 或恢复，并立即返回；本步不会顺带执行后继状态。下一步又先进行全局状态竞争，强召回可立即重新进入 REVIEW。

保存得分满足：

$$
z_{review}-\max(z_{micro},z_{macro})
=1.896981-1.046460
\approx 0.850521 > \Delta_{preempt}=0.15.
$$

因此，仅提高 `state_min_age` 或普通 `state_switch_margin` 无法阻止该抢占。恢复状态也不受保护，且抢占条件未将恢复得分纳入比较，可能造成挂起工作迟迟无法恢复。`SUSPEND_RESUME` 映射到兼容状态 `LEARN_MICRO`，进一步掩盖了真实执行情况。

另一个入口不一致问题是：普通竞争也能选中 REVIEW，但该路径没有统一设置目标和挂起上下文。因此，单独提高 `theta_review_preempt` 只能限制抢占入口，不能关闭所有 REVIEW 入口。

#### 2.2 特征共现代理被当成结构置信度，且归一化抵消成熟度抑制

位置：`_semantic_recall_map()` L1741、`_normalize_map()` L2419、`_projection_nodes_for_optimizer()` L3837。

当前召回场对每个语义关联的 GposI 响应取均值，再对语义取最大值，没有使用关系边的位置约束；所有 GmemI 节点都可以参与投影。它表达的是“这里出现了某些相似的底层特征”，不能直接代表“这里出现了该空间结构”。当前工作语义及未完成语义也参与全局召回。

未完成语义虽乘以 0.5，但最终又进行峰值归一化。只有一个语义且其响应非零时，在有效区域包含峰值的情况下：

$$
R(q)=\frac{0.5\bar S_h(q)}{\max_u 0.5\bar S_h(u)}
=\frac{\bar S_h(q)}{\max_u\bar S_h(u)}.
$$

成熟度折扣完全抵消；微弱的非零响应同样可能产生接近 1 的峰值。将 `theta_review_preempt` 从 0.45 提高到 0.9，仍挡不住日志中的 0.997843。

此外，“窗口外”只表示当前工作窗口外，不表示另一个对象或新证据。大块连续面上的两个相似位置均可位于小窗口外，导致当前工作产生的特征反过来抢占自身学习。

#### 2.3 验证标准过宽，成功后未形成重复验证抑制

位置：`_bfs_recognize()` L3069、`_activate_review_semantic()` L3850、`_run_review_verify()` L3868。

`_bfs_recognize()` 只要发现一个局部特征属于某个语义，就把该语义加入“已识别”集合。它不检查多个特征的空间关系，不要求语义已完成，也不验证是否为本次目标对应的语义；多候选时直接从集合中取一个。即使 `k_depth=0`，直接关联的单特征仍会使验证成功。

成功后 `review_recent_fail` 清零，且没有记录该语义实例刚刚验证过。`Alg_optimizer_beta.md` 第 4.4 节要求近期抑制包含“刚刚失败或刚刚验证过”的结构，当前实现只积累失败。因此，同一局部特征可以反复触发“召回、成功、解除抑制、再次召回”。

提高 `z_review_recent` 或 `review_recent_decay` 对连续成功循环没有作用，因为相应抑制量始终为零。改变神经元不应期也不能直接抑制当前召回代理：召回计算没有读取语义的 `state_flag` 或激活量。

#### 2.4 REVIEW 缺少位置和实例访问抑制，两个峰值可以永久交替

位置：`_peak_outside_window()` L1756、`_run_review_verify()` L3868、`_mark_micro_fixation()` L3247、`_decide_next_saccade_macro()` L3423。

REVIEW 目标是窗口外召回场的确定性最大值，未使用 `ior_map`、`m_aversion` 或近期验证实例记录。在 A 点时排除 A 周围窗口，B 成为峰值；跳到 B 后窗口移动，A 又成为峰值。无论验证成功多少次，这两个候选本身都不被消耗。

新路径的 `m_aversion` 仅在执行 MICRO 向量决策时更新；旧 `_decide_next_saccade_review()` 中的 `ior_map` 更新未接入新 REVIEW。MACRO 虽有语义和完成区域历史抑制，也没有共享的逐次落点抑制。因此“增加旧 REVIEW 的 `alpha_ior`”不会影响当前循环。

#### 2.5 失败量跨工作保留，局部失败会长期压低 MICRO 得分

位置：`_decide_next_saccade_micro_vector()` L2963、`_score_optimizer_states()` L1845、挂起/恢复 L1881/L1903。

`fail_energy` 仅在 MICRO 向量决策中更新，普通跳转、MACRO、REVIEW、恢复或开启另一工作时不会按步衰减或按工作重新绑定。保存值 `fail=1` 会持续给 MICRO 扣 0.9 分、给 MACRO 加 1.1 分；若 MICRO 一直得不到执行，失败量也没有机会衰减。

在同一保存观测下，仅把工作失败量设为零，MICRO 得分将增加至约 1.249，MACRO 降至约 -0.054，但 REVIEW 仍为 1.897。因此失败量残留是放大因素，单独清零不能消除根本的重复召回。

#### 2.6 新路径的完成度指标不可达，完成历史无法正常积累

位置：`SemanticNode.__init__()` L576、`_estimate_completion()` L1771、`_maybe_complete_work()` L3024、旧覆盖计算 L3218。

新语义初始 `is_completed=False`、`boundary_coverage=0`。当前新执行路径没有更新有效的边界覆盖率，相关计算位于未调用的旧 MICRO 场构造方法中。于是新工作即使关系数、子节点数和粗响应项均达到上限，且所有负项为零，也只有：

$$
C_h\leq 0.20+0.15+0.15=0.50<\theta_{complete}=0.72.
$$

`is_completed` 自身还贡献 0.25 的完成度，形成首次完成时不应依赖的条件。该问题不是隔步 REVIEW 的直接触发点，但会阻止新工作正常完成、累计 `semantic_history` 和释放上下文，必须在恢复 MICRO 执行后继续处理。

### 3. 建议的算法修正顺序

1. **统一 REVIEW 候选和进入条件。** 候选至少携带语义编号、候选实例位置、证据版本、结构支持与最近验证记录。普通竞争和抢占应共用同一资格检查及挂起入口；相同实例、相同证据不能因窗口移动再次成为“新召回”。当前未完成工作的自身证据优先支持续学或恢复。
2. **验证后更新实例级短期抑制。** 无论成功或失败均更新，但可使用不同幅度；新证据或结构冲突可解除抑制。抑制应关联语义实例及已验证区域，不能全局封禁同一个底层特征，也不能只抑制单个像素，否则仍会在同一面内换点重复验证。
3. **区分候选发现与验证成功。** GposI 聚合、BFS 只负责快速生成候选；成功需要足够的独立特征与位置关系支持，并绑定本次候选语义。关系置信度应保留绝对尺度，不能逐图拉伸成峰值 1。当前 `_generalized_distance_transform()` L1131 仍是未使用位移参数的均值卷积，不能仅改为调用现有 `l2_structure_synthesis()` 就认为结构验证已经完成。
4. **保证后继工作的执行机会。** 同证据不得立即抢占刚恢复或刚释放的工作；状态年龄应按实际执行步数更新。仍允许具有新证据且价值足够高的其他候选抢占。记录“进入前状态、实际执行状态、结束后状态”，避免把字段切换误当成行为执行。
5. **为所有眼动维护共享的短期位置抑制。** 在统一落点入口更新位置历史，独立于 MICRO 工作访问图；REVIEW 选点同时使用实例抑制。窗口移动后，旧位置的访问记录应保留并随时间衰减。
6. **使失败量属于具体工作，并修复完成度。** 挂起/恢复保存对应工作失败量，新任务重新初始化，跨步适当衰减。完成度应由实际覆盖、关系支持及剩余前沿计算，首次完成不能要求已经完成；不同支持维度只评价适用指标，不能要求所有面或点特征都具有闭合轮廓。

实例抑制可采用以下待实现形式。令 $c$ 为“语义、实例区域、证据版本”候选，$E_c(t)$ 为近期验证量：

$$
E_c(t+1)=\lambda E_c(t)+a_s\mathbf 1[\text{本步验证成功}]
+a_f\mathbf 1[\text{本步验证失败}],\qquad 0<\lambda<1.
$$

$$
U_c(q)=M_{valid}(q)R_c^{structure}(q)
\exp\bigl(-w_cE_c(t)-w_lO_{loc}(q,t)\bigr).
$$

候选选择与 REVIEW 打分共同使用该有效响应，并保留未归一化的结构置信度供绝对阈值判断。不能在抑制后再对每个候选场除以自身峰值，否则会抵消抑制；仅同一证据的重复验证应受限，明确新增的关系证据应触发重新评估。

### 4. 现有参数的分步定位实验

每组实验建议重新创建控制器和记忆图，采用相同输入、初始注视点与步数，避免旧挂起栈及失败量混入。下列改动只用于实验配置，不建议同时全部启用。

| 实验 | 参数 | 解释与限制 |
| --- | --- | --- |
| 暂时排除 REVIEW 的影响 | `state_bias_review = -10.0` | 同时压低普通竞争与抢占的 REVIEW 得分。保存观测下 REVIEW 变为 -8.103，适合判断 MICRO/MACRO 是否能实际执行；这会暂时关闭正常召回，不是最终方案。 |
| 降低召回代理的主导程度 | 恢复默认偏置，设 `z_review_out = 0.5`、`z_review_in = 0.0`、`review_preempt_margin = 0.35` | 保存观测下 REVIEW 变为 0.499，低于 MACRO 1.046。窗内匹配暂不用于发起 REVIEW；仍可能在其他观测下重复抢占，不能替代实例抑制。 |
| 检查完成出口 | 在 REVIEW 问题隔离后，临时设 `theta_complete = 0.45` | 仅用于检查出口代码是否可达；还必须满足前沿、新颖度和缺口门限，不能保证正常完成，更不能替代覆盖率指标修复。 |
| 检查真正的 MACRO 近距离偏好 | 确认实际执行 MACRO 后，将 `gamma_macro_dist` 从 0.001 降至 0.0002，或短暂设为 0 | 只影响 MACRO，不能改变 REVIEW 两峰循环。距离惩罚减小也不等于已加入访问抑制。 |

不建议首先调整：`state_min_age`、`state_switch_margin`（抢占绕过）；`z_review_recent`（成功清零）；`alpha_ior`、`gamma_dist`（新 REVIEW 未使用）；`theta_vector_norm`（仅 MICRO 执行时生效）；`state_temperature`（当前未使用）。单独将抢占阈值提高到 1 以上也不能阻止普通竞争进入 REVIEW。

### 5. 最小调试记录

应在每步同时记录 `debug_optimizer['state']`（处理方法执行前选定状态）与 `behavior_state.name`（结束状态），并加入以下字段：

- `current_state_scores`、`current_state_observation`、`state_age`。
- `review_target` 的进入前与结束后值；后续应增加候选语义编号、验证结果和切换原因。
- `active_semantic_id`、挂起栈长度、`micro_explore_time`、`fail_energy`、`review_recent_fail`。
- 本步起止注视点，区分停留与移动，并统计实际 MICRO 执行次数及关键点写入次数。

修正是否有效，应首先看真实 MICRO 执行机会、重复候选的抑制和新关系增长，而不只是看状态名称是否不再交替。原图完整运行及参数效果仍需结合新的逐步日志验证。


## 二进制特征组合粗筛审查（2026-09-18）

本节为现有代码审查与候选方案，尚未实现或测量新的加速效果。依据 `src/nns/memorygraphs/graph_memorypool_onceoptimizer.py` 的 `HypergraphIndex`、`HierarchyRetriever.query`、`_region_match` 及 `SpatialStructureMatcher.evaluate`。

### 1. 结论与苹果示例

共享特征字典上的二进制编码适合做高层图节点的检索索引，不需要改变 Gmem/Gpos 的主体结构。位为 1 表示相应特征类别可能存在；命中只能产生候选，随后仍须验证连续相似度、成员角色及空间关系。

例如字典为 `[红色区域, 绿色区域, 近圆闭合轮廓, 长条轮廓]`，苹果模板可编码为 `[1,0,1,0]`。输入检测到红色区域和近圆轮廓后，以两类特征的倒排表召回包含该组合的实体，再检查红色区域与轮廓是否属于同一位置、范围是否相容等。红色圆球也可能通过该粗筛，这是允许的假阳性。图左侧有红色方块、右侧有绿色圆盘也会使整图两位同时为 1，因此还需要局部窗口或位姿桶，不能把整图共现等同于同一物体。

如果红色物体有遮挡、光照偏移，或者轮廓不完整，检测器可能只激活一位；硬性要求两位全有就会漏掉相似苹果。反过来，若查询还包含背景的绿色位，也不应要求模板与查询的完整位向量相等。

“圆形”涉及多个边缘位置及方向的组合，不是 CNN 某个局部边缘向量自带的完整语义。当前可由 GmemII 的轮廓区域或新增的闭合度、曲率分布、长宽比等描述子产生候选；“红色区域”可由区域颜色描述子产生。这样的描述子需要跨样本归类及版本管理，并且不能预先假设现有网络已经获得这些语义。描述子用于安全排除前，还必须证明现有精验接受的模板不会被它排除；否则它只是近似排序信号。

### 2. 现有代码已经实现的部分

| 代码位置 | 当前实际行为 | 与期望的差别 |
| --- | --- | --- |
| `FeatureResponseCache.signature` | 以模态、原型形状、原型数值字节和 mask 字节构成签名 | 精确相等归类，不是颜色/轮廓的相似聚类 |
| `HypergraphIndex.__init__` | 相同签名分配同一 class；GmemII 用 `1 << class_id` 组成 `region_bits`，同时建立 class 到模板槽位/偏移的倒排表 | 特征稍有不同即可能占不同位，字典可能随实例增加 |
| `entity_bits / entity_parents` | GmemIII 的位代表成员 GmemII semantic ID，并建立 semantic ID 到实体角色/偏移的倒排表 | 不同 ID 的相似红色区或相似轮廓不会自动成为同一位 |
| `feature_events` | 对每个已有特征 class 计算输入特征相似度，采样点超过 recall 阈值则产生 class、位置、分数事件 | 查询匹配不是字节相等；但产生编码仍需遍历已有特征类 |
| `_votes` | 事件沿倒排表传播，用 `事件位置 - 模板角色偏移` 估计锚点并分位姿桶；统计命中位、角色和分数 | 已经使用位置一致性，不是单纯整图 AND |
| `_votes / _limit` | 以命中不同类别位数/模板类别位数筛选，默认最低比例 0.1；再按覆盖、角色数、分数排序并限额 | 不按最终精验的槽位证据权重计数，不能作为精验的安全必要条件 |
| `HierarchyRetriever.query` | GmemII 候选精验的弱响应继续作为 GmemIII 事件，重复倒排投票与精验 | 已有分层二进制候选机制，并非完全缺失 |
| `_region_match` | 使用 `prior.regions` 中已接受结果作为稀疏中心；仍遍历全部同模态区域模板，无合格稀疏结果就逐区域像素回退 | 位图尚未承担可靠排除整个模板的职责，所以主要瓶颈没有消失 |

GmemII 位为低层精确描述符类，GmemIII 位为 GmemII ID，两层位空间不同，不能直接对它们做 AND。索引在查询时重建，class ID 也不是稳定的长期语义字典。实体位图会合并重复成员 ID，但倒排记录仍保留不同角色；仅看 bit_count 会丢失重复次数。

默认全局事件 stride=4、每类最多 64 个事件、每节点最多 8 个位姿、总候选预算 128。采样、限额、位姿分桶及最低位覆盖都可能遗漏候选。代码还明确说明 `sampled_evaluated_bits` 只代表采样点上做过评估，不能证明该特征在全图不存在。

### 3. “粗筛未找到，一定没见过”何时成立

先把“见过”收窄为可判定定义：**当前记忆快照中，是否存在按指定观测范围、变换范围及精验规则可接受的模板**。它不等同于语义上从未见过该物体，更不能涵盖已被剪枝/遗忘的历史模板。

令 `Accept(t,q)` 表示模板 t 对输入 q 通过既定精验，`Keep(t,q)` 表示粗筛保留它。安全粗筛必须满足

\[
 \operatorname{Accept}(t,q)\Rightarrow\operatorname{Keep}(t,q).
\]

只有满足这个条件，且索引覆盖所有当前模板、查询编码的相关证据搜索完整，才能由“全部模板均被排除”推出“当前快照没有满足该精验规则的匹配”。命中允许是假阳性，未命中不允许是假阴性。这是对粗筛的要求，不是二进制编码天然具备的属性。

若每个模板有必须出现的特征集合 M，查询完整检测到的集合为 Q，并且规则严格要求这些成员全部可见，那么 `(M & Q) == M` 可作为必要条件。其前提是已证明“精验接受必然检测到每个 M 中的位”。当前精验允许部分不可评估和部分槽位未命中，不能把全部模板位都视为必需位；全位包含测试与现有规则不兼容。

以下情况都只能返回“未决/未召回”，不能证明新颖：邻近颜色被单桶量化分开；闭合轮廓检测漏检；遮挡或缺失模态；stride 跳过证据；top-k 或预算截断；模板索引版本陈旧；实例尺度变化超出检测器处理范围。当前代码存在其中多种近似机制，因此其位图未命中不提供确定新颖性证据。

### 4. 与现有容错精验兼容的安全排除判据

建议区分三态：已出现、已充分检查而确定不可能出现、未知。实现可用两张位图，例如“可能出现”与“已认证缺失”；未知必须按可能存在处理。两张普通“命中/检查过”位图本身仍不够，检查过程必须完整且判据保守。

下面给出针对当前覆盖率规则的一种充分排除条件。模板槽位归一化权重为 w_i，总和为 1。令 p_i=1 表示槽位 i 在本次区域与允许位置搜索内可能达到 `graph_recall_threshold`；只有能证明处处达不到时才令 p_i=0。未知槽位令 p_i=1。最终可评估量为 E，命中权重为 H；当前接受规则要求

\[
 E\ge e_{min},\quad H/E\ge c_{min}.
\]

又有 `H <= sum_i w_i p_i`，所以

\[
 \sum_i w_i p_i < e_{min}c_{min}
 \quad\Longrightarrow\quad\text{可安全排除模板}.
\]

当前阈值均为 0.7，乘积为 0.49；安全判断应留浮点保守余量，等于边界时保留。此条件较宽松，但无需预先计算每个中心的实际 E。若已经确定某中心的 E，则可使用更紧的 `sum_i w_i k_i p_i < c_min E`。也可结合现有渐进分数上界继续收紧。

关键是 p_i 的取得必须便宜而且保守：对特征空间桶保存范围摘要，计算输入桶与节点原型的相似度上界；只有所有可能桶的上界都小于 recall 阈值，才认证该槽位缺失。上界必须覆盖实际使用的通道掩码、模态度量及采样规则。边界特征应激活所有相容桶，不能仅取一个最近桶。对于响应双线性插值和位移惩罚，可利用“插值不超过邻接响应最大值、惩罚不超过 1”构造宽松上界，但需要包含所有相关邻接像素，不能只检查原 stride 采样点。

如果同一特征类有多个角色，仍按各槽位权重计入乐观可能量；即使一个事件不能同时满足多个角色，也先保留到精验处理。这会增加假阳性，但避免粗筛误删。普通 popcount/Jaccard/Hamming 距离无法直接代替上述加权覆盖条件。

### 5. 推荐接入方式与实际加速条件

保留现有超图和精验，增加独立的 `CoarseFeatureIndex` 候选索引，分两种使用模式：

1. **高召回近似模式**：区域颜色/轮廓摘要、多桶激活、倒排表、局部位姿投票产生优先候选；其余候选暂不精验时返回搜索不完整。可以先实现并测收益，但不能用未命中直接创建新概念。
2. **保守排除模式**：为节点/桶提供相似度上界及完整性标志，用前述权重必要条件排除不可能模板；只对无法排除的模板运行现有 `_region_match` 位置搜索。未经认证的描述子只影响排序，不参与排除。

GmemII 层可用低层颜色、方向等兼容特征类别筛选区域模板；GmemIII 层可用相似区域类别筛选实体。圆形轮廓作为区域结构描述更适合第二级，不宜以它代替所有底层局部特征。红色区和轮廓应在局部位姿上共同支持实体候选；Gpos 精验最终检查排布。

将相似 GmemII 归入共享类只改变辅助索引，不合并原节点、角色、边和证据；类别 ID 与索引需要版本化，成员更新、原型变化、节点删除/合并后同步维护。硬性多位相交只用于真正必需且已可靠检测的特征，容错模式使用倒排联合召回和加权可能性判断。找到一个候选后仍要保留可能影响歧义 margin 的竞争模板。

预期收益主要来自减少“区域×模板×像素×槽位”精验组合，而非整数 AND 本身。若生成位图仍像当前 `feature_events` 一样遍历所有记忆原型，或者最终仍无条件密集回退，则新增编码可能只增加开销。应一次提取查询摘要，用共享桶/范围索引找到相容类，按命中类访问倒排；常见红色等类别的倒排表可能很长，可先用选择性更高的角色排序，但不能未经证明就删掉低排序结果。

设模板数 N，位数 n，机器字宽 b，单模板精验成本 V，筛后候选数 K。线性扫描位图的成本约为 `O(N ceil(n/b))`，倒排检索则取决于被访问 posting 总量。粗筛有效的条件是

`编码成本 + 索引访问成本 + K*V < N*V`。

不是所有数据都能保证 K 很小；特征过粗会使候选接近全体，特征过细会提高近似编码的漏召回风险。若用固定长度哈希压缩位空间，碰撞可以容许为额外候选，但不能把哈希相等当类别相同；也不能把精确集合查询的无漏检性质直接推广到视觉相似检索。

### 6. 验证计划与本次判断

在冻结记忆、相同输入/阈值/变换范围下，记录编码耗时、posting 访问量、粗筛保留率、后续密集中心数、总延迟、显存与匹配差异。比较候选时应包括能影响模板歧义的竞争项，而不仅比较最终第一名。

用颜色桶边界、同色不同形、同形不同色、红色与圆形空间分离、局部遮挡、缺失模态、重复角色和强弱权重不均等样本审查漏筛。对保守模式需要数学上的上界保证和实现测试；样本测试零漏检本身不能证明永不漏检。小规模密集参考也只能验证当前精验规则的一致性，现有槽位赋值不回溯，不能当作所有可能图匹配的数学完备判定器。

本次判断：**方案有效且与现有结构兼容；代码已有部分机制，但特征类别语义、编码生成成本及区域匹配入口仍需改进。当前不能由位图未命中断言“没见过”。要获得可用于新颖性判断的否定结果，应采用保守兼容编码、完整性状态和与精验一致的必要条件；其余情况保持未决。**


## 增长图的索引、并行与分类读出讨论（2026-09-18）

根据最新 38 图运行，本次方案审查详见 [报告](../record/2026-09-18_index_parallelism_growing_graph_classifier.md)。粗编码允许多模板共享候选桶；自适应分裂必须使用查询可检测的特征，并对跨边界或未知输入保留多分支。优先消除关系全表扫描及无关通道造成的索引差异，再在同一步冻结快照上并行匹配、确定性归并和串行提交。分类先建立稳定 ID 映射的可扩展稀疏线性基线，长期采用共享结构编码和固定维度集合读出；节点新增、合并、删除及内容版本变化均须进入训练回放与快照管理。上述为待实施方案，未修改运行算法。


## 标注框监督与判别闭环（2026-09-18）

新增方案见 [Alg_SupervisedGraphLearning.md](Alg_SupervisedGraphLearning.md)。有效标注实例可直接建立受保护的 annotation_confirmed GmemIII，真实视觉支持不伪增；框内成员仍允许纠正，类别与实体模板一对多。优先完成已知框分类，使用固定类别证据池化适应节点增长，再补充无框候选、Gpos 框定位与检测评价。视频与光流后续以带时间对应和置信度的对象观察接入，不能把非零运动直接当语义前景。当前为待实施方案。
