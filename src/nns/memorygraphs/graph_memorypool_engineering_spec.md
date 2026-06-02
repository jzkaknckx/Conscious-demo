
# `graph_memorypool.py` 核心算法变动

---

### 一、 视觉前端 (CNN 特征呈递)

**这部分由其他小组实现，你无需关心CNN特征呈递改动的原理，但需要注意特征呈递的数据结构。特征张量的每个子空间传来的通道数不固定**

输入张量 $X = [X_{hue}, X_{edge}, X_{curv}, X_{texture}]$，在逻辑上被显式正交切分为独立的子空间。

* **现有模态数据：**
* 明暗对比与边缘信息。
* 局部空间频率与纹理（基于 SWT 算法）。
* 方向与局部曲率（构建几何形状的基元）。


* **（待实现）色彩拮抗通道 (Color Opponency)：** 引入红/绿（R-G）和蓝/黄（B-Y）通道以替代或补充单纯的色相（Hue），提升模型在不同环境下的色彩恒常性分辨能力。

---

### 二、 Gmemory (静态拓扑记忆网络)

负责长时拓扑与特征固化。该网络抛弃反向传播，由注视点扫视行为驱动节点生长与拓扑记录。

#### 1. GI 层：模态孤立特征层

* **数据结构：** 废弃全通道融合。每个节点独立且轻量，仅包含：
* 模态标识符 (Modality ID)。
* 低维子空间原型向量 $\mathbf{w}^{(m)} \in \mathbb{R}^{c_m}$。



#### 2. GII 层：锚点-外围拓扑层 (语义对象)

* **数据结构：** 存储由一个 Anchor 和多个 Peripheral 组成的集合。包含锚点特征 $\mathbf{w}_{anc}$ 与外围特征集 $\{\mathbf{w}_1, \dots, \mathbf{w}_N\}$。
* **（待实现）锚点选举算法 (Anchor Election)：** 在扫视学习间隙，根据特征连通域范围和激活频次（count）自动选举具有最高覆盖度或稳定性的 GI 节点作为 Anchor，确立其余特征为 Peripheral。
* **（待实现）极坐标拓扑编码：** 计算并记录外围节点相对于锚点的理想对数极坐标偏移向量 $\{(\rho_1, \theta_1), \dots, (\rho_N, \theta_N)\}$。
* **（待实现）形变惩罚系数：** 记录每个 Peripheral 允许的空间游离容忍度参数 $\{(\lambda_1, \gamma_1), \dots, (\lambda_N, \gamma_N)\}$。

---

### 三、 （待实现）Gposition (动态投影检索网络)

自身无固定学习参数，完全接收 Gmemory 注入的算子进行全图并行的结构匹配。

#### 1. （待实现）L1 层：模态路由子空间投影

* **（待实现）算法：** 接收 GmemI 的特征请求。复用器（Multiplexer）根据 Modality ID 对输入场 $X$ 进行硬切片，仅在对应的低维子空间内执行 1x1 卷积（点积）。
* **（待实现）输出数据：** 生成极高信噪比的纯模态相似度得分图，包括锚点得分 $S_{anc}(x,y)$ 与各外围特征得分 $S_n(x,y)$。

#### 2. （待实现）L2 层：结构合成与连续尺度解算

* **（待实现）锚点候选锁定算法 (Anchor Proposal)：** 在 $S_{anc}(x,y)$ 上执行局部非极大值抑制（NMS），提取高分候选点坐标 $(x_c, y_c)$。
* **（待实现）对数极坐标扭曲 (Spatial Warping)：** 以候选点 $(x_c, y_c)$ 为物理原点，将对应的笛卡尔矩阵 $S_n(x,y)$ 重采样映射为对数极坐标张量 $\mathbf{S}_{LP, n}(\rho, \theta)$。将物体的尺度缩放（乘法）转换为 $\rho$ 轴平移（加法）。
* **（待实现）广义距离变换算法 (Generalized Distance Transform)：** 基于 GmemII 提供的理想偏移 $\{(\rho_n, \theta_n)\}$ 与惩罚系数 $(\lambda_n, \gamma_n)$，对 $\mathbf{S}_{LP, n}$ 求解允许形变的最高得分图：

$$D_n(\Delta\rho, \Delta\theta) = \max_{\delta\rho, \delta\theta} \left[ \mathbf{S}_{LP, n}(\rho_n + \Delta\rho + \delta\rho, \theta_n + \Delta\theta + \delta\theta) - \lambda_n(\delta\rho)^2 - \gamma_n(\delta\theta)^2 \right]$$


* **（待实现）全局拓扑共振解算：** 将锚点基础得分与所有 $D_n$ 矩阵相加得到 $G_{pos}^{II}$。通过 $\arg\max$ 反解出精确的连续尺度放大率 $s^* = e^{\Delta\rho^*}$ 与整体旋转角 $\Delta\theta^*$。

---

### 四、 核心交互链路 (双塔联动)

* **（待实现）Top-Down 结构注入：** `Gmemory` 将特征原型（外观）和相对拓扑与容忍度（结构）作为临时计算图参数，并行下发给 `Gposition` 对应层级。
* **（待实现）跨模态归一化与自适应加权：** 在 `Gposition` 执行“全局拓扑共振”相加合并各子模态相似度得分前，引入量纲对齐策略，防止尖锐分布的模态（如边缘）掩盖平缓分布的模态（如大面积色相）。
* **（待实现）Bottom-Up 兴趣反馈：** `Gposition` 产出的最终全局相似度矩阵 $S_{map}$ 与决策器的动态兴趣矩阵点乘，对相似部分抬高兴趣值，引导下一次寻找和形状沿边缘的扫视学习。





---

# `graph_memorypool.py` 重构技术规范 (v2.0)

## 1. 架构升级概述

本次重构的核心目标是将原有的“早期融合、全通道稠密匹配”的单塔图网络，升级为“特征/拓扑解耦、晚期融合、连续尺度自适应”的双塔架构。
系统将严格划分为两大核心模块：

* **Gmemory (记忆塔)**：静态存储网络，负责记录独立的模态特征与以 Anchor-Peripheral 为核心的极坐标空间拓扑。
* **Gposition (检索塔)**：动态计算网络，无自身训练参数，接收 Gmemory 的拓扑参数，在 CNN 的密集特征场上执行 1x1 子空间投影、对数极坐标扭曲 (Log-Polar Warping) 与广义距离变换 (GDT)。

---

## 2. 核心名词与变量映射表

为确保算法推演与代码实现的一致性，请研发团队参照下表进行命名和变量替换：

| 算法概念 (Algorithm Concept) | 对应代码变量名 / 结构 (Code Variable / Struct) | 备注与类型 |
| --- | --- | --- |
| **模态子空间 (Modality Subspace)** | `X_subspaces: Dict[int, torch.Tensor]` | 切分后的稠密 CNN 特征场，如 `0: hue`, `1: edge` |
| **色彩拮抗 (Color Opponency)** | `rg_channel`, `by_channel` | 待实现的视觉基元，补充 `hue` |
| **模态孤立节点 (GI Node)** | `ModalityNode` (替代 `Graph1Proto`) | 存储单模态低维原型向量 `w_m` |
| **语义对象节点 (GII Node)** | `SemanticNode` (替代 `Graph2Node`) | 存储 Anchor-Peripheral 拓扑 |
| **核心锚点 (Anchor)** | `anchor_id: int` | 指向一个 `ModalityNode` (通常是大面积连通域特征) |
| **外围特征 (Peripheral)** | `peripheral_links: Dict[int, Dict]` | Key: peripheral_id, Value: 拓扑约束参数字典 |
| **对数极坐标偏移 (LP Offset)** | `rho_offset`, `theta_offset` | 在 `peripheral_links` 中，理想的 $\rho, \theta$ 相对值 |
| **形变惩罚系数 (Deformation Penalty)** | `lambda_rho`, `gamma_theta` | 约束外围特征游离容忍度的二次项系数 |
| **局部极坐标图 (LP Warped Map)** | `S_LP: torch.Tensor` | $S_n(x,y)$ 经坐标扭曲后的张量 |
| **广义距离变换得分 (GDT Map)** | `D_map: torch.Tensor` | 应用形变惩罚后的最高得分图 |
| **全局相似度矩阵 (Global S_map)** | `Gpos_II_map: torch.Tensor` | 融合所有子模态与拓扑约束后的最终矩阵 |

---

## 3. 数据结构修改规范 (Gmemory 模块)

目标文件中的 `Graph0Node`, `Graph1Proto`, `Graph2Node` 数据结构将被废弃，替换为以下新结构。

### 3.1 重构 GI 层：模态专用节点 `ModalityNode`

原 `GraphCollection` 采用稠密 `bitset` 编码全模态，现将其拆分为按模态隔离的存储结构。

* **新增 Class: `ModalityNode**`
* `modality_id: int` (标识该节点属于哪个子空间，例如 0=边缘，1=色相，2=RG拮抗等)
* `prototype: torch.Tensor` (低维子空间特征向量，替代原 `signature` 和 `bitset`)
* `count: int`, `last_seen: int` (保留，用于后续生命周期管理)


* **改动说明：** 原有的 `GraphI` 类更名为 `GmemoryI`，内部的字典从按 Bucket 索引，改为首先按 `modality_id` 划分独立的 Node 库。

### 3.2 重构 GII 层：锚点-外围节点 `SemanticNode`

废弃原 `Graph2Node` 中的 `members` 集合和笛卡尔 `centroids`，改用星型拓扑图记录物体结构。

* **新增 Class: `SemanticNode**`
* `node_id: int`
* `anchor_id: int` (指向 `GmemoryI` 中的核心特征节点)
* `peripheral_links: Dict[int, Dict[str, float]]`
* 数据格式要求：`{ peri_id: {'rho_offset': float, 'theta_offset': float, 'lambda_rho': float, 'gamma_theta': float} }`




* **交互逻辑说明：**
* **（待实现）锚点选举 (Anchor Election)：** 在 `batch_update_from_hypothesis` 期间，不再无脑 Merge。需评估扫视捕获的所有特征，选择连通域最大（基于 SWT 响应或 `texture_resp`）且 `count` 最高的节点设为 Anchor。
* 计算其余节点相对 Anchor 质心的对数极坐标 $(\rho, \theta)$ 并存入 `peripheral_links`。



---

## 4. 算法与流水线修改规范 (Gposition 模块)

这是本次重构的重心。原有的基于 `bitset` 检索的 `GraphI.fast_update` 和基于质心 RBF 的 `GraphII.soft_merge` 被彻底替换为 **Gposition 连续稠密张量流**。

### 4.1 L1 层：模态路由子空间投影 (Modality-Routed Subspace Projection)

* **输入：** 原始 `extract_features` 放弃提取 Sparse Top-K 点，改为将 CNN 呈递的特征硬切分为稠密张量字典：`X_subspaces = {0: X_edge, 1: X_hue, 2: X_curv...}`。
* **算法：** `Gposition` 接收 `Gmemory` 注入的 `ModalityNode` 的 `prototype`。
* 充当多路复用器（Multiplexer），基于 `modality_id` 选中对应的 `X_subspaces[id]`。
* 执行 $1 \times 1$ 卷积操作：$S_m(x,y) = \mathbf{w}^{(m)} \cdot X_m(x,y)$。


* **输出：** 单模态的相似度张量集合 $S_{anc}(x,y)$ (锚点得分) 和 $\{S_n(x,y)\}$ (各外围得分)。

### 4.2 L2 层：结构合成与对数极坐标匹配 (Structure Synthesis)

新增 `GpositionII` 运算模块，执行以下 Pipeline：

1. **锚点候选锁定 (Anchor Proposal):**
在 $S_{anc}(x,y)$ 上执行 `F.max_pool2d` 提取局部极值，获取一系列候选物理原点 $(x_c, y_c)$。
2. **对数极坐标扭曲 (Log-Polar Warping):**
* 围绕每个 $(x_c, y_c)$，使用 `F.grid_sample`（或自定义的坐标映射算子）将笛卡尔张量 $\{S_n(x,y)\}$ 重采样为极坐标张量 $\mathbf{S}_{LP, n}(\rho, \theta)$。
* **原理提醒：** 这一步将现实世界物体的缩放（乘法）严格转换为了 $\rho$ 轴的平移（加法）。


3. **广义距离变换 (GDT):**
引入 `Gmemory` 提供的结构先验参数。对每一张 $\mathbf{S}_{LP, n}$，计算其形变后的最高得分图 $D_n$：

$$D_n(\Delta\rho, \Delta\theta) = \max_{\delta\rho, \delta\theta} \left[ \mathbf{S}_{LP, n}(\rho_n + \Delta\rho + \delta\rho, \dots) - \lambda_n(\delta\rho)^2 - \gamma_n(\delta\theta)^2 \right]$$


4. **全局拓扑共振解算:**
* 合并得分：$Gpos\_II\_map(\Delta\rho, \Delta\theta) = S_{anc}(x_c, y_c) + \sum D_n(\Delta\rho, \Delta\theta)$。
* （待实现）**跨模态归一化：** 在相加前，对 $D_n$ 乘以基于模态方差计算的动态权重，避免边缘的尖锐峰值淹没色相的平缓峰值。
* 反解最优解：通过 `torch.argmax` 获取峰值所在的偏移量 $(\Delta\rho^*, \Delta\theta^*)$。
* 提取真实物理尺度与角度：放大率 $s^* = e^{\Delta\rho^*}$，旋转角 $\theta^* = \Delta\theta^*$。



---

## 5. 跨模块数据交互与决策整合 (Multilevel Coordinator)

`MultilevelCoordinator` 和 `AttentionPolicy` 的行为需要适配双塔输出。

1. **Bottom-Up 兴趣反馈 (Gpos -> AttentionPolicy):**
* `Gposition` 输出的全局相似度峰值结果，需要反映射回原图像的笛卡尔坐标系中，生成一张稠密的全局匹配热力图。
* 该热力图直接与 `AttentionPolicy` 中的动态兴趣矩阵（`interest_map`）进行**点乘 (Hadamard Product)** 或 **加权求和**。
* **目标：** 对于匹配度高的区域，强制抬高其周围的局部边缘兴趣值，引导 `Saccade` 模块移动注视点进行后续的学习和形状验证。


2. **Top-Down 结构注入 (Gmem -> Gpos):**
* 实现一个专用的 `compile_to_gpos_operators()` 方法。当 `Gmemory` 接收到识别请求时，能够将其静态节点快速打包为 $1 \times 1$ 卷积权重（用于 GposI）和相对坐标惩罚矩阵（用于 GposII）。



---

## 6. 配置项 `MemoryConfig` 增补清单

在原有的 `MemoryConfig` 类中，需新增以下超参数以支持新算法的平滑运行：

```python
class MemoryConfig:
    # ... 保留原有的基本配置 ...
    
    # 极坐标采样配置 (Log-Polar Config)
    lp_rho_bins = 64        # 极径采样精度
    lp_theta_bins = 64      # 极角采样精度
    lp_base_radius = 2.0    # 对数极坐标的基础起算半径
    
    # 广义距离变换形变约束 (GDT Constraints)
    default_lambda_rho = 1.5   # 默认的尺度游离惩罚系数
    default_gamma_theta = 2.0  # 默认的角度游离惩罚系数
    
    # 模态权重平衡 (Modality Normalization)
    modality_weights = {
        0: 0.4,  # edge
        1: 0.6,  # hue
        2: 0.5,  # RG/BY (待实现)
        # ...

    # 补齐其他所需超参数
    }

```