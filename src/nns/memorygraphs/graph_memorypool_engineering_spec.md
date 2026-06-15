# 语义掩码 (Semantic Mask) 生成算法方案探讨

针对“寻找一个连续的纯色区域（如苹果内部及其外围边缘），同时拒绝外侧具有相似特征但物理不连通的噪点或其他物体”的目标，采用基于张量化形态学的拓扑连通域扩散生成语义掩码。

### 基于张量化形态学的拓扑连通域扩散 (Tensor-based Topologic Diffusion)

为了提取严格连通的纯色域和过渡边缘，我们使用**“拓扑路径连通度”代替单纯的“直线距离”**。在基于 PyTorch 张量体系中，可基于最大池化与系数控制执行极其快速的“张量域漫水扩散”（Tensor Dilation/Diffusion）。

#### 1. 算法定式

**步骤 1：生成全局通透率图 (Permeability Map)**
利用 Gpos 相似度生成面域响应 $M_{resp\_con}$，并将原始的基础底场强度 $I_{base}$ 中不属于面连续性的高频强突变点（如非此颜色的锋利划痕）视作屏障（阻力）：
$$P_{map}(\mathbf{q}) = M_{resp\_con}(\mathbf{q}) \cdot \operatorname{Sigmoid}(1 - \lambda \cdot I_{str}(\mathbf{q}))$$
它代表了流体在底图上的穿行许可率。若此时与起跳点同质，通过率 $\approx 1$；若为异质对象或强墙壁，通过率 $\approx 0$。

**步骤 2：初始化种子核心 (Seed Initialization)**
建立一个全 0 掩码张量，仅在注视锚点中心 $\mathbf{p}_{macro}$ 点燃核心：
$$V_0(\mathbf{p}_{macro}) = 1.0, \quad V_0(\mathbf{q} \neq \mathbf{p}_{macro}) = 0.0$$

**步骤 3：张量蔓延迭代 (Iterative Matrix Dilation)**
利用最大池化算子（`MaxPool2d(kernel_size=3, stride=1, padding=1)`），每一帧向外“膨胀”一圈，但每次膨胀后必须再与 $P_{map}$ 逐元素相乘。这确保了掩码在每一次外扩时都会受到相似度边界的严格阻挡。
迭代 $T$ 步（使得膨胀能覆盖假设中的最大可能特征域，或差值收敛）：
$$V_{t+1} = \text{MaxPool2d}(V_t) \odot P_{map}$$

**步骤 4：生成归一化掩码 (Final Mask Output)**
蔓延终止后，由于池化的降权和相乘的滤波，形成一片中间高两边自然衰减的水坑，它完美适配物体真形。采用平滑锐化出图：
$$M_{semantic} = \operatorname{Sigmoid}(\gamma \cdot V_T - \delta)$$

#### 2. 卓越表现对照
- **天然切割孤立杂质**：即使附近有一个与之颜色一样的噪点，只要这处噪点与核心之间，隔了一条“其他颜色或强对比环境的沟壑”，迭代扩散就**根本无法翻越**那条导致 $P_{map} \to 0$ 的鸿沟，完美根除了外侧噪点粘附现象。
- **精确吸附连续边缘**：在同质面域边缘处，$P_{map}$ 将伴随相似度的下降呈现滑坡衰减。扩散引擎会向外爬坡衰弱并天然静止在外围的半坡地带，将连续面量与自身附着的轮廓边缘极其漂亮地卷入一个张量中。
- **算力相容开销极小**：相较传统的 CPU 基于像素/图论的深搜泛滥算法，这个操作完美映射到 GPU 的 Conv2D 核下。$T=20$ 或 $30$ 次的 Pooling+乘法迭代，只需耗时不到 1~2 毫秒，且具备 100% 的张量梯度连贯形式。


# graph_memorypool.py 视知觉重构技术规格书 (V6.0)

## 0. 概述 (Overview)
针对特征跨越物体边界“拉丝”与原地停留的死锁，必须基于通道解构将控制器改迁为 **Macro-Micro 双轨控制模型**，同时将“空间同质性”进行更加细粒度的降维打散：分别进行**表面颜料面评价**与**骨骼走线方向评价**以获得精纯无噪的 $I_{map}$。

---

## 1. 深度属性分拣：CONTINUITY 面域与线域的独立拆分解析

为了从根源上剔除非对象同质特征（避免噪声导致的掩码和注意力外泄），必须对原先粗暴的 `CONTINUITY` 进行解囊，剥离维数特性。

- **[修改点 1.1: 特征属性配置映射表重构]**
  构建更为精细的字典映射：`cfg.feature_attributes[mod_idx][channel_idx]`。枚举合法值为：`STRENGTH`（绝对强弱量）、`CONTINUITY_SURFACE`（面量，如明度光影或Hue）、`CONTINUITY_TRACE`（线量，如Orientation方向走势）以及 `IGNORE`。
- **[修改点 1.2: Base Interest 与 GmemI 门限动态投递核审]**
  - **STRENGTH**: 
    - 基础场提供: 纯振幅绝对拉抬 $I_{str} \leftarrow \sum X_c$
    - GmemI收录核审: 绝对值 $\mathbf{v}_c > \tau_{\text{str\_gating}}$。
  - **CONTINUITY_SURFACE**:
    - 基础场评价: 无方向各向同性方差评估 $val_{surf} = \exp(-\|\nabla X_c\|^2 / 2\sigma_{surf}^2)$
    - GmemI收录核审: 取点周边一致形变分 $val_{surf} > \tau_{\text{surf\_gating}}$。
  - **CONTINUITY_TRACE**:
    - 基础场评价: 取样片区内单位矢量走向相干聚合度 $val_{trace} = \frac{\|\text{Blur}(\vec{v}_{c})\|}{\text{Blur}(\|\vec{v}_{c}\|)}$。
    - GmemI收录核审: 提供片段线段高一致度 $val_{trace} > \tau_{\text{trace\_gating}}$ 才允准挂载，规避碎叶花点写入。

---

## 2. 宏观/微观双轨切换控制与纯化特征域生成的软掩码约束

废弃粗放型判断。引入 $M_{semantic}$ 注意力界限软光圈，实现从粗落点寻找物体再到辖区内部细瞄的双流程控制。

- **[修改点 2.1: Semantic Mask 由且仅由 CONTINUITY_SURFACE 高纯供给生成]**
  - **触发**: 在 Macro 大步跨越后，成功降落起锚生成了新 Anchor 节点时刻。
  - **执行提取**: 遍历此刻为该原点位置被激发的初盘 GmemI Active 节点集合，**极其严格地筛除掉一切不是 `CONTINUITY_SURFACE` 类的特征对象**。
  - **场回波重构与拓扑扩散生成掩码**:
    1. **通透率图生成 (Permeability Map)**：利用 Gpos 提取由上述纯面特征生成的响应度 $M_{resp\_con}$，结合非面连续性（如强度边缘）突变点作为环境阻力网格构建阻值：$P_{map}(\mathbf{q}) = M_{resp\_con}(\mathbf{q}) \cdot \operatorname{Sigmoid}(1 - \lambda \cdot I_{str}(\mathbf{q}))$。
    2. **张量蔓延迭代 (Iterative Matrix Dilation)**：以注视锚点中心 $\mathbf{p}_{macro}$ 创建单值源初值 $V_0(\mathbf{p}_{macro})=1.0$ 的零矩阵。利用底层可导的连续极大约束算子执行 $T$ 步膨胀控制：$V_{t+1} = \text{MaxPool2d}(V_t) \odot P_{map}$。
    3. 最终蔓延水坑经过平滑化 $M_{semantic} = \operatorname{Sigmoid}(\gamma \cdot V_T - \delta)$ 输送为崭新掩码态载入 Controller。
- **[修改点 2.2: Micro-Saccade 封锁态 (局部深耕)]**
  - **锁入机制**: 上个周期生成掩码及记录 `active_semantic_id`，立即锁转 Micro 引擎。
  - **场力斩截**: 生成出来的全图兴趣场必须通过 $M_{semantic}$ 阿尔法相乘：$I_{masked} = I_{map} \odot M_{semantic}$。从而绝育周边界限上的所有其余环境 $I_{base}$ 及引导力召唤。
  - **辖区罚距折算**: 将该软掩码非零域面积近似视作面积 $S$，推演出等效圆半径 $R_{scale}$，从而动态折返配用小粒度的视跨越跳行衰减距离 $\sigma_{micro} \propto R_{scale}$。使得跳步永远紧缩在本域。
- **[修改点 2.3: 陨坑满溢与退行破溃 (Breakout)]**
  - **判定阈值退行**: Micro 眼跳下经过的锚区将受到带有 $-1$ 不应权的空间双调制（大感受野正小极坑负）疯狂摧残，留下全是一路死寂的连串坑洞。当该区域被踏平使全区残值 $\max(I_{masked}) < \epsilon$ 时，标志当前物体已无法剥削。
  - **破笼重生**: 丢弃清空 $M_{semantic}$ 矩阵，$active\_semantic\_id \leftarrow \text{None}$，彻底解绑该 GmemII 组。放归为 Macro，跳眼被未踏足区域庞大的全新 Base Interest 远吸走。
