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
* **（新增算法）锚点选举算法 (Anchor Election)：** 在扫视学习间隙，根据特征连通域范围和激活频次（count）自动选举具有最高覆盖度或稳定性的 GI 节点作为 Anchor，确立其余特征为 Peripheral。
* **（新增算法）极坐标拓扑编码：** 计算并记录外围节点相对于锚点的理想对数极坐标偏移向量 $\{(\rho_1, \theta_1), \dots, (\rho_N, \theta_N)\}$。引入全局度量尺 `scaler`，对所有新记录的极径进行归一化：$\rho_{mem} = \ln(\|\vec{v}_{pixel}\| / \text{scaler})$。
* **（新增算法）形变惩罚系数：** 记录每个 Peripheral 允许的空间游离容忍度参数 $\{(\lambda_1, \gamma_1), \dots, (\lambda_N, \gamma_N)\}$。

#### 3. （新增算法）Gmemory 拓扑演化与修正规则

* **（新增算法）稳定性倒置：** 持续评估特征的主从关系。若 Peripheral 的稳定性得分超越其 Anchor，触发坐标系倒置，原 Peripheral 升格为 Anchor，拓扑向量取逆。
* **（新增算法）菊花状拓扑退化：** 针对高频复现的深层树状结构（如 $A \to B \to C$），通过笛卡尔坐标过渡计算跨级偏移量，将结构直接压缩为星型（菊花状）拓扑，以消除嵌套检索的复杂度。
* **（新增算法）畸变纠正与新事物分支：** 当 GposII 匹配发生空间偏差，计算形变能量 $E$。若 $E \le \text{阈值}$，利用指数移动平均（EMA）微调原有拓扑偏移量；若 $E > \text{阈值}$，则克隆原 Anchor 生成变种分支。

---

### 三、 （待新增算法）Gposition (动态投影检索网络)

自身无固定学习参数，完全接收 Gmemory 注入的算子进行全图并行的结构匹配。

#### 1. （新增算法）L1 层：模态路由子空间投影

* **（新增算法）算法：** 接收 GmemI 的特征请求。复用器（Multiplexer）根据 Modality ID 对输入场 $X$ 进行硬切片，仅在对应的低维子空间内执行 1x1 卷积（点积）。
* **（新增算法）输出数据：** 生成极高信噪比的纯模态相似度得分图，包括锚点得分 $S_{anc}(x,y)$ 与各外围特征得分 $S_n(x,y)$。

#### 2. （新增算法）L2 层：结构合成与连续尺度解算

* **（新增算法）锚点候选锁定算法 (Anchor Proposal)：** 在 $S_{anc}(x,y)$ 上执行局部非极大值抑制（NMS），提取高分候选点坐标 $(x_c, y_c)$。
* **（新增算法）对数极坐标扭曲 (Spatial Warping)：** 以候选点 $(x_c, y_c)$ 为物理原点，将对应的笛卡尔矩阵 $S_n(x,y)$ 重采样映射为对数极坐标张量 $\mathbf{S}_{LP, n}(\rho, \theta)$。将物体的尺度缩放（乘法）转换为 $\rho$ 轴平移（加法）。
* **（新增算法）广义距离变换算法 (Generalized Distance Transform)：** 基于 GmemII 提供的理想偏移 $\{(\rho_n, \theta_n)\}$ 与惩罚系数 $(\lambda_n, \gamma_n)$，对 $\mathbf{S}_{LP, n}$ 求解允许形变的最高得分图：

$$D_n(\Delta\rho, \Delta\theta) = \max_{\delta\rho, \delta\theta} \left[ \mathbf{S}_{LP, n}(\rho_n + \Delta\rho + \delta\rho, \theta_n + \Delta\theta + \delta\theta) - \lambda_n(\delta\rho)^2 - \gamma_n(\delta\theta)^2 \right]$$


* **（新增算法）全局拓扑共振解算：** 将锚点基础得分与所有 $D_n$ 矩阵相加得到 $G_{pos}^{II}$。通过 $\arg\max$ 反解出精确的连续尺度放大率 $s^* = e^{\Delta\rho^*}$ 与整体旋转角 $\Delta\theta^*$。

---

### 四、 核心交互链路 (双塔联动)

* **（新增算法）Top-Down 结构注入：** `Gmemory` 将特征原型（外观）和相对拓扑与容忍度（结构）作为临时计算图参数，并行下发给 `Gposition` 对应层级。
* **（待新增算法）跨模态归一化与自适应加权：** 在 `Gposition` 执行“全局拓扑共振”相加合并各子模态相似度得分前，引入量纲对齐策略，防止尖锐分布的模态（如边缘）掩盖平缓分布的模态（如大面积色相）。
* **（新增算法）Bottom-Up 兴趣反馈：** `Gposition` 产出的最终全局相似度矩阵 $S_{map}$ 与决策器的动态兴趣矩阵点乘，对相似部分抬高兴趣值，引导下一次寻找和形状沿边缘的扫视学习。

---

### 五、 （新增算法）控制器 (Controller) 状态机与眼跳规则

完全替换旧的扫描逻辑，改用基于动态兴趣图的三态生物学状态机控制眼跳与学习。

* **（新增算法）全局度量尺交互：** 控制器维护全局变量 `scaler`。每当 GposII 成功检出目标并输出放大率 $s^*$ 时，将其同步更新至 `scaler`。
* **（新增算法）兴趣图 (Interest Map) 动态维护：** 基于边缘与纹理生成基础层，叠加轨迹抑制 (M_trace) 与永久抑制 (M_comp)，并接受 GposII 投射的局部高斯峰值作为 Top-Down 反馈调制。
* **（新增算法）三态眼跳状态机：**
1. **全局探索 (Explore)：** 注视点向兴趣图的全局绝对最大值处做长程跳跃，提取区域特征以建立新的 Anchor。
2. **局部审查 (Inspect)：** 锁定 Anchor 后，兴趣图产生中心辐射效应，注视点沿梯度方向向外跳跃，提取特征建立 Peripheral。
3. **巩固校验 (Consolidate)：** 基于 GposII 对某缺失部件的物理位置预测，在兴趣图该位置注入强制高斯峰，制导眼跳过去验证特征，完成闭环。



---

---

# `graph_memorypool.py` 重构技术规范 (v2.0)

## 1. 架构升级概述

本次重构的核心目标是将原有的“早期融合、全通道稠密匹配”的单塔图网络，升级为“特征/拓扑解耦、晚期融合、连续尺度自适应”的双塔架构，并引入生物学状态机接管视觉扫视策略。
系统将严格划分为三大核心模块：

* **Gmemory (记忆塔)**：静态存储网络，负责记录独立的模态特征与以 Anchor-Peripheral 为核心的极坐标空间拓扑，维护拓扑的动态演化。
* **Gposition (检索塔)**：动态计算网络，无自身训练参数，接收 Gmemory 的拓扑参数，在 CNN 的密集特征场上执行 1x1 子空间投影、对数极坐标扭曲 (Log-Polar Warping) 与广义距离变换 (GDT)。
* **Controller (控制器)**：取代旧的扫视逻辑，基于动态兴趣矩阵维护三态状态机（探索/审查/校验），调配双塔交互并控制物理注视点移动。

---

## 2. 核心名词与变量映射表

为确保算法推演与代码实现的一致性，请研发团队参照下表进行命名和变量替换：

| 算法概念 (Algorithm Concept) | 对应代码变量名 / 结构 (Code Variable / Struct) | 备注与类型 |
| --- | --- | --- |
| **模态子空间 (Modality Subspace)** | `X_subspaces: Dict[int, torch.Tensor]` | 切分后的稠密 CNN 特征场，如 `0: hue`, `1: edge` |
| **色彩拮抗 (Color Opponency)** | `rg_channel`, `by_channel` | 待实现的视觉基元，补充 `hue` |
| **模态孤立节点 (GI Node)** | `ModalityNode` (替代 `Graph1Proto`) | 存储单模态低维原型向量 `w_m` |
| **语义对象节点 (GII Node)** | `SemanticNode` (替代 `Graph2Node`) | 存储 Anchor-Peripheral 拓扑 |
| **核心锚点 (Anchor)** | `anchor_id: int` | 指向一个 `ModalityNode` |
| **外围特征 (Peripheral)** | `peripheral_links: Dict[int, Dict]` | Key: peripheral_id, Value: 拓扑约束参数字典 |
| **稳定性得分 (Stability Score)** | `stability_score: float` | 用于主客体倒置的动态评分 |
| **对数极坐标偏移 (LP Offset)** | `rho_offset`, `theta_offset` | 理想的 $\rho, \theta$ 相对值 |
| **广义距离变换得分 (GDT Map)** | `D_map: torch.Tensor` | 应用形变惩罚后的最高得分图 |
| **动态度量尺 (Dynamic Scaler)** | `global_scaler: float` | 统一全局尺度的乘积因子 |
| **状态机枚举 (State Enum)** | `SaccadeState: EXPLORE, INSPECT, CONSOLIDATE` | 替换原 `saccade.state` 逻辑 |

---

## 3. 数据结构修改规范 (Gmemory 模块)

目标文件中的 `Graph0Node`, `Graph1Proto`, `Graph2Node` 数据结构将被废弃，替换为以下新结构。

### 3.1 重构 GI 层：模态专用节点 `ModalityNode`

原 `GraphCollection` 采用稠密 `bitset` 编码全模态，现将其拆分为按模态隔离的存储结构。

* **新增 Class: `ModalityNode**`
* `modality_id: int` (标识该节点属于哪个子空间，例如 0=边缘，1=色相，2=RG拮抗等)
* `prototype: torch.Tensor` (低维子空间特征向量，替代原 `signature` 和 `bitset`)
* `count: int`, `last_seen: int`, `stability_score: float`


* **改动说明：** 原有的 `GraphI` 类更名为 `GmemoryI`，内部的字典从按 Bucket 索引，改为首先按 `modality_id` 划分独立的 Node 库。

### 3.2 重构 GII 层：锚点-外围节点 `SemanticNode`

废弃原 `Graph2Node` 中的 `members` 集合和笛卡尔 `centroids`，改用星型拓扑图记录物体结构。

* **新增 Class: `SemanticNode**`
* `node_id: int`
* `anchor_id: int`
* `peripheral_links: Dict[int, Dict[str, float]]`
* 数据格式要求：`{ peri_id: {'rho_offset': float, 'theta_offset': float, 'lambda_rho': float, 'gamma_theta': float} }`



### 3.3 （未实现）Gmemory 演化机制实现规范

在 `GmemoryII` 模块内部新增维护拓扑动态生长的逻辑函数：

* **主客倒置规则：** 定期比对 `peripheral_links` 中的 `peri_id` 指向节点的 `stability_score` 与 `anchor_id` 的得分。若触发倒置，执行：$\theta_{new} = (\theta_{old} + \pi) \pmod{2\pi}$。
* **路径压缩转换：** 针对 $A \to B \to C$ 关系，提取 $r_{AB} = e^{\rho_{AB}}, r_{BC} = e^{\rho_{BC}}$。在笛卡尔系下计算 $\Delta x, \Delta y$ 后，重组为 $\rho_{AC} = \ln(\sqrt{\Delta x^2 + \Delta y^2}), \theta_{AC} = \arctan2(\Delta y, \Delta x)$ 并挂载。
* **微调与分支：** 设置阈值 `deformation_threshold`。匹配回传的误差 $E \le$ 阈值时，使用 `ema_alpha` 更新 `rho_offset` 和 `theta_offset`。

---

## 4. 算法与流水线修改规范 (Gposition 模块)

原有的基于 `bitset` 检索的 `GraphI.fast_update` 和基于质心 RBF 的 `GraphII.soft_merge` 被彻底替换为 **Gposition 连续稠密张量流**。

### 4.1 L1 层：模态路由子空间投影

* **输入：** 原始 `extract_features` 放弃提取 Sparse Top-K 点，改为将 CNN 呈递的特征硬切分为稠密张量字典：`X_subspaces = {0: X_edge, 1: X_hue, ...}`。
* **算法：** 接收 `ModalityNode` 的 `prototype`。充当多路复用器，执行子空间 1x1 卷积：$S_m(x,y) = \mathbf{w}^{(m)} \cdot X_m(x,y)$。

### 4.2 L2 层：结构合成与对数极坐标匹配

1. **锚点候选锁定:** 在 $S_{anc}(x,y)$ 上执行 `F.max_pool2d` 提取局部极值 $(x_c, y_c)$。
2. **对数极坐标扭曲:** 围绕 $(x_c, y_c)$，使用 `F.grid_sample` 将笛卡尔张量 $\{S_n(x,y)\}$ 重采样为极坐标张量 $\mathbf{S}_{LP, n}(\rho, \theta)$。
3. **广义距离变换 (GDT):**
引入 `Gmemory` 先验参数，对每一张 $\mathbf{S}_{LP, n}$ 计算形变惩罚后的最高得分图 $D_n$：

$$D_n(\Delta\rho, \Delta\theta) = \max_{\delta\rho, \delta\theta} \left[ \mathbf{S}_{LP, n}(\rho_n + \Delta\rho + \delta\rho, \dots) - \lambda_n(\delta\rho)^2 - \gamma_n(\delta\theta)^2 \right]$$


4. **全局拓扑共振解算:**
合并得分 $Gpos\_II\_map(\Delta\rho, \Delta\theta) = S_{anc}(x_c, y_c) + \sum w_n D_n(\Delta\rho, \Delta\theta)$。
获取最优偏移量 $(\Delta\rho^*, \Delta\theta^*)$，提取物理放大率 $s^* = e^{\Delta\rho^*}$ 同步给全局 `global_scaler`。

---

## 5. 跨模块数据交互与控制器 (Controller) 状态机

替换原 `EpisodeManager` 与 `AttentionPolicy` 的核心循环，改由新控制器负责：

1. **Bottom-Up 兴趣反馈与状态转移：**
* **EXPLORE (探索)：** 平滑全局 `interest_map`，寻找绝对最大值点触发长程眼跳。到达后提取 Anchor 特征。
* **INSPECT (审查)：** GposII 反射匹配结果至 `interest_map` 形成中心辐射高地。眼跳沿梯度向外移动以构建 Peripheral 关系（计算偏移时必须除以当前的 `global_scaler`）。
* **CONSOLIDATE (校验)：** 接收 GposII 传来的缺失特征预测坐标。在 `interest_map` 强制注入高斯峰制导眼跳，验证后更新 Gmem 拓扑或创建新分支。


2. **Top-Down 结构注入：**
* 实现专用方法 `compile_to_gpos_operators()`。当 `Gmemory` 响应识别时，将静态节点打包为 GposI 的 1x1 卷积权重和 GposII 的相对坐标/惩罚系数矩阵。



---

## 6. 配置项 `MemoryConfig` 增补清单

在原有的 `MemoryConfig` 类中，需新增以下超参数以支持新算法：

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
    }

    # --- 新增：拓扑演化与控制器参数 ---
    global_scaler_init = 1.0       # 动态度量尺初始值
    ema_alpha = 0.05               # 畸变微调更新率
    deformation_threshold = 2.5    # 触发新事物分支的形变容忍度阈值
    saccade_explore_sigma = 15.0   # 探索状态下的高斯平滑核参数
    saccade_consolidate_gain = 5.0 # 校验状态下预测点兴趣值的注入增益

```