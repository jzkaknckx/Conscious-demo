# 视觉大模型特征表征与相似度度量探讨 (Representation Learning & Similarity Metrics)

本文档记录了关于底层特征提取、1x1 卷积操作在特征测度上的数学原理及其核心局限性，以及针对“属性类”（如方位角、色相）特征的全新相似度度量与重构方案。

## 一、1x1 卷积原理与其在表征域的限制

在早期的神经网路特征架构中，主要依靠内积操作（如 1x1 卷积、余弦相似度）度量局部感受到输入张量 $X$ 与目标原型向量 $W$ 之间的匹配度。然而从数学与代数物理的视角解析，该机制对其表征空间的维度性质有极其严苛的限定。

### 1. 适用领域：占有度变量 (Magnitude Variables)
若向量所属的高维空间表征的是某个概念方向上的“占有度”或“强度大小”（例如基于 $dx$ 和 $dy$ 表达的图像一阶空间梯度），那么这样的多维向量在欧式空间中有确切的方向与模长。
此时，利用 1x1 卷积（即对应通道数值乘加）来计算相互关联具备极佳的代数合法性：实质上它等价于在测量输入特征落在此原型向量方向上的投影绝对有效值或二者夹角。对于这一类**包含强度几何描述的结构化特征空间**，传统余弦与内积累加机制高度适用。

### 2. 限制与失效领域：属性变量 (Property Variables)
诸如“边缘朝向（Orientation）”、“色相（Hue）”等特征完全不具备欧式线性几何性：它们是闭环的周期性标量参数或是绝对定义的类别标签。它们的数值（如 $5^\circ$ 或 $355^\circ$、$Hue_{red}$ 或 $Hue_{blue}$）仅用以标记质的差异定性，而不具备线性加减和投影的物理容忍度。
*   **底层物理意义崩溃**：强制将两个表征绝对角度大小的标量直接内积或作为通道值做 1x1 卷积，产出的代数积和在数学上毫无意义。
*   **生理学认知差异**：在人类等高等生物视觉皮层（V1/V2区）内，对于朝向与色彩的响应是由相互离散、具有极强特异性的功能柱（如方位柱、颜色柱）共同承担的。它们以高度类似**One-hot 放电编码**的离散激发态实现信息传递。
*   **维度爆炸灾难**：如果为了适配卷积算子而尝试使用巨型的完全型 One-hot 向量来表示所有离散色彩与极小间隔的方位，将带来极其恐怖的通道暴增问题（通道上万），使得整个系统的显存管理及图拓扑计算体系过载破溃。

因此，在一个特征混合维（哪怕只包含了 0 = grad_Intensity, 1 = grad_Orientation）上直接简单粗暴应用任何点积卷积，寻找“边缘的准确相似图景”是注定错误且无效的技术误途。

---

## 二、属性类（类 One-hot）特征相似度求解方法

为了解决以上困局、避开高维 One-hot 表征暴增的资源代价，必须对单纯表征定性的属性类维度专门设计拓扑匹配手段。以下确立三种层层递进或互补的计算重构方案：

### 方案 A：周期距离度量与高斯 RBF 核屏蔽 (Cyclic Distance & RBF Kernel)
**核心思想**：彻底舍弃线性内积。针对具备明确物理周期性的拓扑特性（例如首尾相连的色环角 $0 \to 2\pi$，或者完全对称的灰度梯度朝向 $0 \to \pi$），手动引入环路最短差异距离计算。借助高斯核的快速跌落特性制作绝对硬隔离的响应掩膜。

*   **计算模型**：
    设视场当前的属性张量为 $O_X(\mathbf{q})$，记忆原型目标为 $o_w$，周期最大跨度为 $P$。
    1.  计算点对环域绝对差值：$\Delta O(\mathbf{q}) = \min \left(|O_X(\mathbf{q}) - o_w|, P - |O_X(\mathbf{q}) - o_w|\right)$
    2.  带入高斯径向基核进行投影：$S_{prop}(\mathbf{q}) = \exp \left( - \frac{\Delta O(\mathbf{q})^2}{2\sigma^2} \right)$
*   **计算优势**：高度贴近常理思维，允许直接操控 $\sigma$ 参数设定特异性“容忍范围”（例如控制边缘夹角误差只能在 $\pm 5^\circ$），完美压制全区域底噪泛化。

### 方案 B：连续欧几里得展开与高阶三角内积 (Trigonometric Embedding / Continuous One-hot Equivalent)
**核心思想**：若底层引擎难以放弃基于并行的 1x1 矩阵点积计算的高并发红利，则可以利用极少量通道增广将标量属性强制降维推演至合法可解的连续化欧式空间之中，这是兼顾 One-hot 的隔离度与点积计算速度的超维操作。

*   **计算模型**：
    将单通道属性特征角 $\theta$ 展开转化为具有两个通道的单位圆正交映射结构：
    $$ \mathbf{V}_{property}(\mathbf{q}) = [\cos(k\theta_X(\mathbf{q})), \sin(k\theta_X(\mathbf{q}))] $$
    （其中角系数 $k$ 的选择取决于对称性：若是边缘差分 $180^\circ$ 和 $0^\circ$ 一样，则取 $k=2$；若是针对全环色相颜色，则取 $k=1$）。
    在匹配时，调用通用的 1x1 卷积点积操作计算，得到这实质上是余弦定理展开结果的未修正项：
    $$ S_{raw} = \mathbf{V}_{X} \cdot \mathbf{V}_{W} = \cos \left( k(\theta_X - \theta_W) \right) $$
    随后叠加非线性高次幂进行锐化收口（极大增加其对细微角度差的敏感拒绝度）：
    $$ S_{prop} = \left( \max(0, S_{raw}) \right)^{\gamma} $$
*   **计算优势**：架构破坏力极小！极大保留了现有卷积网络框架在正向推理上带来的硬件红利，仅额外使用1通道便达成“类 One-hot 匹配效果”。

### 方案 C：主强门控与特征解耦双规并行 (Intensity Gating & Property Decoupling)
**核心思想**：效法于高级生理解剖，从网络流向底层进行分离。提取模块出具的张量应当从一开始分离并彻底隔绝为“存在性能量”（Intensity / Magnitude）与“属性能量”（Angle / Hue）。两者异构分别验算后进行相乘联觉。

*   **计算模型**：
    1.  **强度底噪扫描**：特征仅依赖于其几何结构大小进行底噪投射：$S_{int}(\mathbf{q}) = X_{int}(\mathbf{q}) \odot W_{int}$。这决定了某处“是否有边界”。
    2.  **属性极性过滤**：基于方案 A 或 B 获取对当前定性标量的强拒绝掩模 $S_{prop}(\mathbf{q})$。这决定了某处“它是不是属于所希望的方位边界”。
    3.  **异构耦合关联反馈**：
        两路互不认识的数据结构进行硬隔离相乘叠加：
        $$ S_{final}(\mathbf{q}) = S_{int}(\mathbf{q}) \odot S_{prop}(\mathbf{q}) $$
*   **计算优势**：逻辑梳理最为清晰、层次最为分明，确保了图突触连结投射的基础建立在“目标本身的存在性必然存在（如必须先存在真实边缘梯级）”，此后再叠加它对应的方位容忍性检测。绝对抹杀了任何背景伪类导致的假阳性激活。

## 三、相似性检索原理

在确立了占有度（Magnitude）与属性（Property）变量的差异后，本节从数学角度严谨地论述针对不同变量类型以及混合变量进行相似性检索的底层原理与物理意义。

### 1. 占有度变量（多通道 $C_m > 1$）使用 $1\times 1$ 卷积检索相似度的原理

多通道的占有度变量（如基于笛卡尔坐标的梯度强度 $[dx, dy]$、多频段的滤波器响应强度等）能够张成一个具备真实几何意义的特征向量空间（Vector Space）。
在这样的欧几里得空间中，$1\times 1$ 卷积操作在数学上完全等价于通道维度的点积（Dot Product）：
$$S_{mag}(\mathbf{q}) = \sum_{c=1}^{C_m} X_{c}(\mathbf{q}) \cdot W_{c} = \mathbf{X}(\mathbf{q}) \cdot \mathbf{W}$$
**原理剖析：**
点积的几何意义是 $\mathbf{X}$ 在 $\mathbf{W}$ 方向上的投影长度乘以 $\mathbf{W}$ 的模长。当特征向量被赋予了“强度大小”和“方向”复合表达时，内积正好反映了：输入信号在目标原型特征维度上的能量一致性。它不仅捕获了分量分布的相似程度（通过余弦夹角），同时也奖励了绝对信号强度（模长叠加）。因此，在这类变量子空间内的 $1\times 1$ 卷积是一次严密且合法的欧式几何相似度映射。

### 2. 属性变量经过单位圆正交映射结构检索相似度的原理

属性变量（如色相、绝对朝向角度）本质上是一个标量环（Manifold, 例如 $S^1$ 圆周），不具备线性空间的加减法合法性。直接对其标量值应用卷积会陷入荒谬的代数错误。
引入单位圆正交映射结构（即上述的“连续欧几里得展开”），则是将 1D 的流形强行嵌入到 2D 欧氏空间中的一种非线性基底变换（Basis Embedding）：
$$ \mathbf{V}_{prop}(\theta) = [\cos(k\theta), \sin(k\theta)]^T $$
在此正交编码下的点积计算：
$$ S_{raw} = \mathbf{V}_X(\mathbf{q}) \cdot \mathbf{V}_W = \cos(k\theta_X)\cos(k\theta_W) + \sin(k\theta_X)\sin(k\theta_W) = \cos(k(\theta_X - \theta_W)) $$
**原理剖析：**
经过单位圆映射后，原本失去物理意义的点积（$1\times 1$ 卷积），通过三角恒等变换，精妙地转化为了计算两个标量属性在特征环上的角距离余弦。随后引入高阶锐化 $S_{prop} = \left( \max(0, S_{raw}) \right)^\gamma$，就相当于在这个相空间中塑造了一个极其尖锐的、类似于 Dirac-delta 函数或高斯激活的感受野（Receptive Field），完美达成了类 One-hot 的强排他性相似度度量。

### 3. 混合变量作 Hadamard 乘积相似度的论证与数学意义

在部分视觉特征中（例如提取的 `edge` 模态），往往同时解耦出了“几何强度（如梯度的绝对幅值，属于占有度）”和“边缘朝向（如梯度的角度，属于属性）”。此前的方案 C 提出将这两路的检出响应做 Hadamard 乘积 $S_{final} = S_{int} \odot S_{prop}$ 来作为最终的特征图激活表达。表面上看，这似乎用乘法偏离了传统卷积网路的“内积与加和”体系，但其具备极其生动且严密的理论合法性，可从以下四个数学与物理视角进行论证：

*   **视角一：条件概率与贝叶斯推断视角 (Conditional Probability & Bayes)**
    在概率空间中，$S_{int}$ 可以视为在局部区域存在某种特征基底的先验或边际概率估值 $P(\text{存在边缘})$；而 $S_{prop}$ 是在已经确定存在边界的前提下，其属性完全符合预期目标的条件概率 $P(\text{朝向} = \theta_W \mid \text{存在边缘})$。两路响应相乘即是联合概率密度：
    $$ P(\text{目标出现}) = P(\text{存在边缘}) \cdot P(\text{朝向} = \theta_W \mid \text{存在边缘}) $$
    因此，乘法在此处是联合概率事件同时发生的最完备表达。

*   **视角二：模糊逻辑的 T 范数视角 (Fuzzy Logic T-norm)**
    若将特征的检出视作对真值域在 $[0,1]$ 的模糊命题求交集：$S_{int}$ 表示命题 A（“这里有一个突变的坎”），$S_{prop}$ 表示命题 B（“这个坎的方向是水平的”）。我们真正在寻找的是复合命题 $A \land B$。在模糊逻辑代数（Fuzzy Set Theory）中，用于定义 AND 操作的标准的代数 T-norm 恰恰就是代数乘积：
    $$ \mu_{A \land B}(x) = \mu_A(x) \cdot \mu_B(x) $$
    这种相乘严格且平滑地反映了多重认知条件的全满足要求。

*   **视角三：门控与信息流调制视角 (Gating & Modulation)**
    在循环神经网络（如 LSTM/GRU）模型设计中非常普遍的一种结构叫作“门控流”。$S_{prop}$ 被视作一个流形上的状态阀门（Gate），其范围映射至 $[0,1]$；而 $S_{int}$ 是承载底噪的信息能源量或通量。乘法激活机制 $S_{int} \odot S_{prop}$ 等价于通过“属性相似度”作为门控，动态导通或削弱原本基层传递的信号能量。这种非线性的强度衰减机制能极其有效地切断那些边缘鲜朗却方位彻底迥异（假阳性）的噪音传导。

*   **视角四：物理能量相干检波机制 (Coherent Detection)**
    将神经网络内的特征响应波视为一种连续发射的电磁波，$S_{int}$ 代表了包络线的振幅（Amplitude Volume），而 $S_{prop}$ 则表征了空间载波的相位干涉相干度（Phase Coherence）。在诸如雷达检波或外差接收机体系中，最终解离出的有效能量输出，永远是绝对振幅被接收机本振与外发差信号之间的相位差混频乘波（等效于相乘削峰幅）调合后的结果。不共相的振动互相抵消而截断（输出逼临 0），唯有强度高且同频共相的区域激发出高阶包络共振。

综上所述，剥离粗糙混合态进而将隔离通道异构相乘的双轨并发设计，不仅成功规避了单纯依靠 $1\times 1$ 卷积陷入代数困境的陷阱，更是同时向贝叶斯理论、信道调变科学和模糊判定等深层数学体系实现了完美借壳落地，赋予了这套新型视觉系统更具生命演化色彩的联觉基础原理支撑。

---

## 四、Gpos 位置相似度检索原理审查

本节审查 Gpos 各层是否已经具备“位置相似度”检索能力，并给出应有的数学形式。这里的“位置相似度”不是指单个特征在图像中的响应位置，而是指：当若干特征节点共同出现时，它们的**相对位置矢量结构**是否与 Gmem 中记忆的边结构相似。

### 1. 图结构边的多重性与下一层节点存储规范

当前图结构的基本设定是：节点代表特征，节点之间的边代表特征间的位置矢量，下一层节点代表上一层节点集合及其内部关系。因此，当两个相同特征原型 $A,B$ 以不同空间组合出现时，例如第一次为左右排布、第二次为上下排布，不应把它们压缩为同一条边。

设两个观测实例分别给出：
$$
e_1=(A \rightarrow B, \Delta \mathbf{p}_1), \quad
e_2=(A \rightarrow B, \Delta \mathbf{p}_2)
$$
若
$$
d_{\Sigma}(\Delta \mathbf{p}_1,\Delta \mathbf{p}_2)
=
(\Delta \mathbf{p}_1-\Delta \mathbf{p}_2)^T
\Sigma^{-1}
(\Delta \mathbf{p}_1-\Delta \mathbf{p}_2)
> \theta_{edge}
$$
则二者必须作为两条不同的**关系边实例**或两个不同的**关系模式**保存。原因是下一层节点的判别对象不是 $\{A,B\}$ 这个无序集合，而是带结构的子图：
$$
G^{(2)} = (V^{(1)}, E^{(1)})
$$
若只保存端点集合，则左右结构与上下结构会被错误地视为同一语义组件。

因此，下一层节点至少应保存边的唯一身份或关系槽位，而不能只保存目标节点 id。推荐的数据语义为：
$$
r_j = (edge\_id, src\_id, dst\_id, \mu_{\Delta}, \Sigma_{\Delta}, count, confidence)
$$
其中 $\mu_{\Delta}$ 是相对位置均值，$\Sigma_{\Delta}$ 是位置容忍协方差或等价的 $(\lambda_\rho,\gamma_\theta)$ 容忍参数。一个 GmemII 节点应保存：
$$
SemanticNode = \left(
\{child\_node\_id\},
\{relation\_edge\_id\},
anchor\_id
\right)
$$
而不是简单的 `target_id -> edge_value` 字典。

这也解释了当前实现中的结构风险：`SemanticNode.peripheral_links` 以 `peri_id` 为 key，`EntityNode.components` 以 `sem_id` 为 key。若同一目标特征或同一语义组件在同一父节点中出现多个位置模式，后写入的边会覆盖先前边。正确形式应当是多重图（Multigraph）或“目标 id 到边列表”的结构：
$$
links[(src,dst)] = \{e_1,e_2,\dots,e_K\}
$$
只有当新观测边与某个已有边模式满足 $d_{\Sigma}<\theta_{edge}$ 时才更新该边模式；否则创建新的边模式。这样既允许同一对端点形成多种组合，又避免无限重复存储相近样本。

### 2. GposI：具备特征位置响应，不具备结构位置相似度

GposI 的数学功能是把单个 GmemI 原型 $\mathbf{w}_i$ 投影回当前视野，得到该特征在每个位置的响应图：
$$
S_i(\mathbf{q}) = \operatorname{Sim}\left(X_m(\mathbf{q}), \mathbf{w}_i\right)
$$
它回答的是：“哪里像这个底层特征？” 这属于**特征位置响应**，不是结构位置相似度。它不检查 $A$ 与 $B$ 是否保持某个相对位置矢量，只产生后续 GposII 进行结构匹配所需的基础响应场。

从当前代码看，`Gposition.l1_modality_routing_projection` 与 `Gposition.l1_footprint_projection` 已实现这一层能力：它们对每个 `ModalityNode` 调用 `SimilarityEngine.calculate_similarity` 生成整图响应。但这一层没有边结构输入，因此不能单独判定“左右排布”或“上下排布”是否相似。

### 3. GposII：理论上应承担相对位置相似度检索

对一个 GmemII 语义节点，设锚点为 $a$，外围关系边集合为 $\mathcal{E}=\{e_k\}$。每条边记录目标特征 $b_k$ 相对锚点的记忆位移 $\Delta_k=(\rho_k,\theta_k)$ 与容忍核 $\Sigma_k$。对于候选锚点位置 $\mathbf{c}$，GposII 应计算：
$$
S_{II}(\mathbf{c})
=
S_a(\mathbf{c})
\cdot
\prod_{k}
\left[
\max_{\mathbf{q}}
S_{b_k}(\mathbf{q})
\exp\left(
-\frac{1}{2}
d_{LP}(\mathbf{q};\mathbf{c},\Delta_k)^T
\Sigma_k^{-1}
d_{LP}(\mathbf{q};\mathbf{c},\Delta_k)
\right)
\right]
$$
其中 $d_{LP}$ 是在以 $\mathbf{c}$ 为原点的对数极坐标系内，候选目标点 $\mathbf{q}$ 与记忆位移 $\Delta_k$ 的差异。若采用加性对数形式，可写作：
$$
\log S_{II}(\mathbf{c})
=
\log S_a(\mathbf{c})
+
\sum_k
\log D_k(\mathbf{c})
$$
其中
$$
D_k(\mathbf{c})
=
\max_{\mathbf{q}}
S_{b_k}(\mathbf{q})
K_k\left(
LP_{\mathbf{c}}(\mathbf{q})-\Delta_k
\right)
$$
该式证明了 GposII 的核心作用：它不是简单累加多个 GposI 响应，而是在候选锚点坐标系下检查每个外围特征是否落在记忆边所规定的相对位置容忍域内。

若允许旋转与尺度弹性，可进一步引入全局变换参数 $(s,\phi)$：
$$
S_{II}(\mathbf{c},s,\phi)
=
S_a(\mathbf{c})
\cdot
\prod_k
D_k(\mathbf{c},s,\phi)
$$
最终取
$$
S_{II}^{*}(\mathbf{c})
=
\max_{s,\phi} S_{II}(\mathbf{c},s,\phi)
$$
这给出了平移不变、可选尺度/旋转弹性的结构检索形式。其本质是“锚点候选 + 相对边一致性”的匹配，而不是单点特征匹配。

当前代码中 `Gposition.l2_structure_synthesis` 只实现了这一思想的雏形，尚未形成可用的 GposII 位置相似度图：
*   它只选择 `S_anc` 的一个最佳峰作为锚点候选，没有对全图所有候选锚点生成 $S_{II}(\mathbf{c})$。
*   `_generalized_distance_transform` 接收 `rho_offset`、`theta_offset`、`lambda_rho`、`gamma_theta`，但函数体只做 5x5 平滑卷积，没有使用这些参数计算位置偏差惩罚。
*   `l2_structure_synthesis` 在 `run_step` 中没有被调用，因此当前控制器没有真正使用 GposII 的结构位置相似度响应。
*   `Controller._decide_next_saccade_macro` 中的 `sum_I_spatial_II` 只是把语义节点包含的 GposI 响应相加并乘以激活值，它能产生“相关特征曾经出现过”的抑制图，但不能证明这些特征满足记忆中的相对位置结构。

因此结论是：**GposII 在文档设计上应具备位置相似度检索功能，但当前实现尚未真正具备。**

### 4. GposIII：当前仅有记忆结构，没有位置检索实现

GposIII 的设计目标是把多个 GmemII 组件按照 GmemIII 中记录的宏观位姿边进行格式塔检出。理论形式与 GposII 相同，只是节点从 GmemI 特征响应换成 GmemII 结构响应：
$$
S_{III}(\mathbf{c})
=
S_{A}^{II}(\mathbf{c})
\cdot
\prod_j
\left[
\max_{\mathbf{q}}
S_{B_j}^{II}(\mathbf{q})
K_j\left(
LP_{\mathbf{c}}(\mathbf{q})-\Delta_j^{III}
\right)
\right]
$$
其中 $S_{B_j}^{II}$ 是某个局部组件的 GposII 响应图，$\Delta_j^{III}$ 是宏观组件间的位置边。

当前代码中 `GmemoryIII` 只保存 `EntityNode.components`，并提供 `add_component` 写入接口，没有对应的 `Gposition.l3_*` 检索函数。`components` 也以 `sem_id` 为 key，存在与 GmemII 类似的多位置覆盖问题。因而当前实现中 GposIII 不具备位置相似度检索能力。

### 5. 对本次拓扑意图场方案的修正

在 Gpos 审查后，拓扑意图场中各目标/候选场应按以下原则接入 Gpos 响应：

| 候选场或目标 | 需要的 Gpos 响应 | 作用 |
| --- | --- | --- |
| $C_{color}(\mathbf{q})$ | GposI Surface/Color 响应 | 确认当前位置是否仍属于同一内部颜色原型 |
| $M_{\Omega}(\mathbf{q})$ | GposI Surface 响应 + 连通扩散 | 生成当前面域的导通掩码 |
| $B_{\Omega}(\mathbf{q})$ | GposI Boundary/Grad 响应 + $\nabla M_{\Omega}$ | 定位面域与突变边界的交界 |
| $I_{corner}, I_{junction}, I_{terminator}$ | GposI 曲率/方向/边缘响应；可叠加局部几何算子 | 发现高信息量轮廓点 |
| `BOUNDARY_SEEK` | 主要使用 GposI；若已有同类 GmemII 被激活，可叠加 GposII 预测边界先验 | 新对象学习时以底层边界为主，复现识别时可用结构先验 |
| `CONTOUR_TRACE` | GposI 边界响应为主；GposII 边界链响应为辅 | 沿真实边界前进，并在熟悉结构中减少试探 |
| `LEARN_MACRO` 抑制已探索对象 | 应使用真正的 GposII/GposIII 结构响应；当前代码只能用近似的 L1 响应和 `semantic_history` | 防止回到已学习区域，并优先寻找新语义组件 |
| `REVIEW` 记忆验证 | 应优先使用 GposII/GposIII | 验证整图结构是否复现，而不是只靠局部特征命中 |

因此，本次方案需要调整为：**新对象的 MICRO 学习主要依赖 GposI 生成面域、边界、角点候选；已有对象的 REVIEW/MACRO 抑制与预测应依赖真正实现后的 GposII/GposIII 结构响应。** 在 GposII 尚未实现完整位置相似度前，不应把 `M_GposII` 作为已经可靠的结构抑制图使用，只能将其视为待实现接口或临时近似项。
