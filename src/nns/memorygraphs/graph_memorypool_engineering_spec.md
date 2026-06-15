# 视觉大模型认知算法架构：眼跳驱动与图记忆更新

该文档确立了在`REVIEW`, `LEARN_MACRO`, `LEARN_MICRO`三种认知状态下的眼跳驱动机制、特征写入方案以及图神经网络状态的更新逻辑，以解决注意力粘滞与状态机非正常闭环等问题。

## 一、新眼跳驱动策略

为适应不同探索阶段的目的，眼跳驱动模型现解耦为三种独立的策略。

### 1. REVIEW 状态（记忆验证阶段）
*   **驱动目标**：对已有记忆特征（`interest_map`）高响应区域进行复查。
*   **计算模型**：注视点 $p_{next}$ 取决于记忆兴趣值与空间抑制的结合。
    $$p_{next} = \arg\max \left[ I_{map}(\mathbf{q}) - \alpha \cdot IoR(\mathbf{q}) - \gamma \cdot Dist(\mathbf{p}_{curr}, \mathbf{q}) \right]$$
*   **IoR演化**：全局抑制图 $IoR(\mathbf{q})$ 每次注视后在当前点 $\mathbf{p}_{curr}$ 进行高斯叠加并自然衰减，强制打破多次定焦死锁。

### 2. LEARN_MACRO 状态（宏观锚点搜索阶段）
*   **驱动目标**：寻找视觉中大面积连续平坦色块区域，并**规避已经探索学习过的语义区域**。
*   **计算模型**：
    由于原始 $S_{base}$ 多由边缘高频特征主导导致最大值易出现在物体边缘，故改用从神经计算中提取的代表连续色块响应的连续性提取图 $I_{con1}$：
    $$I_{macro}(\mathbf{q}) = I_{con1}(\mathbf{q}) \odot \left(1 - \overline{\sum I_{spatial}^{II}}(\mathbf{q}) \right)$$
    其中 $\overline{\sum I_{spatial}^{II}}$ 为GmemoryII（已经学习到的语义团）空间激活响应对数和的归一化抑制掩码，用于防止眼跳在已建构宏观锚点的区域内打转。
    利用形态学空间低通滤波 $K_{lowpass}$ 进行平滑：
    $$p_{next} = \arg\max \left( I_{macro} * K_{lowpass} \right)$$
    以确保宏观锚点降落在大面积连续色块的内心物理几何中心。

### 3. LEARN_MICRO 状态（微观特征精细扫描阶段）
*   **驱动目标**：在软掩模 $M_{semantic}$ 限制下的辖区内深度扫描。
*   **计算模型**：抛弃全局记忆分布 $I_{map}$，回归无偏的底层特征，且仅对软掩模辖区激发。
    $$p_{next} = \arg\max \left( S_{base}(\mathbf{q}) \odot M_{semantic}(\mathbf{q}) \odot (1 - M_{aversion}(\mathbf{q})) \right)$$
*   **厌恶足迹机制（跳出机制）**：
    初始化生命周期单次 `LEARN_MICRO` 的状态变量 $M_{aversion}$。每次扫描 $p_i$ 后，叠加中央凹尺寸的抑制分布：
    $$M_{aversion}^{(t+1)}(\mathbf{q}) = \max \left( M_{aversion}^{(t)}(\mathbf{q}) , \exp \left( -\frac{||\mathbf{q} - \mathbf{p}_i||^2}{2\sigma_{fovea}^2} \right) \right)$$
*   **退出条件计算**：每次眼跳检查厌恶足迹对当前语义结构的覆盖率指标 $\rho$：
    $$\rho = \frac{\sum (M_{aversion} \odot M_{semantic})}{\sum M_{semantic}}$$
    若 $\rho > \theta_{exit}$，表示该面状区内信息已被吸干，将该 `semantic_mask` 从缓存彻底清空，并将状态机退出为 `LEARN_MACRO`。

## 二、特征写入策略

*   **延迟隔离存储（延期挂载机制）**：`LEARN_MICRO` 过程的所有 Peripheral 特征采集仅推入临时内存队里 `peripheral_buffer` 中。期间不改变大图主存储网络，从而确保 $sum\_I\_spatial$ 等宏观掩模场形依然静置不动。
*   **中心爆发汇聚网络**：微观扫描破笼结束并转移回 `LEARN_MACRO` 时，调用批处理（Batch Processing），将之前产生的核心面特征即 Anchor，作为连接的主干点。网络将队列内 Peripheral 特征集合挂接成隶属该锚点的次级辐射特征星团边。

## 三、tick_update 机制调整

针对局部神经元频繁进入不应期并产生非预期粘滞及由于过高频建图造成的计算灾难，更新机制被解耦：

### 1. 连续微时间动力学（局部状态迭代）
眼跳内部（特别是指代大量微眼跳发生时），**不触发**任何诸如神经元充放电、图网络神经元层级的更新操作，即**彻底挂起 `tick_update`**。由于图电位不发生实质修改，神经元永远不因此陷入不应期（Refractory Period）导致的感受野过饱和破溃。

### 2. 离散宏时间动力学（拓扑级同步刷新）
*   **触发时机**：**仅当系统从 LEARN_MICRO 状态发生跃迁且强制回归到 LEARN_MACRO 时段时**，执行一次全局的 `tick_update` 过程。
*   **状态同步统一结算**：
    1.  执行 Peripheral 向 Anchor 的批次向心挂载。
    2.  全局进行电位消散、激活及不应期的集体统筹。
    3.  系统获得全新的稳定视觉兴趣基底，在下一次 $I_{macro}$ 计算时，旧有的已被探索物体的区域 $\sum I_{spatial}^{II}$ 掩护成功生效，驱动视眼探索全图未知域。
