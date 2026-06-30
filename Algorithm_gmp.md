# 视觉认知与记忆流形重构算法规范

本模块聚焦于 `graph_memorypool.py` 中最基础、原生的数据结构——记忆图谱池（Gmem）与空间动力投影网（Gpos），并详述两者交互的精确数学定义。这是更高阶场驱动与宏微观眼跳引擎运转的地基。

## 一、 分层分布记忆网池 (Gmem: Modality, Semantic & Entity Nodes)

Gmem 采用物理脱钩的多层拓扑设计。底层 `ModalityNode` (GmemI) 直指感知孤立块特征；中层 `SemanticNode` (GmemII) 不保存像素数据本身，仅存取指向对象的指针与极坐标骨架结构（外围拓扑约束）；高层 `EntityNode` (GmemIII) 负责宏观实体建构。一切神经元节点的流形生命周期均服从统一的状态机。

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
GmemII 的建立使纯孤立特征组合成为局部空间物体结构（组件层）：
- **心锚 (Anchor_ID)**: 标记对象的绝对原点（指向一枚在物体中心的 ModalityNode）。
- **极引力挂载树 (Peripheral Links)**:
  对外围所拾获之点 $k$，建立起相对于其主锚 $anc$ 空间之极坐标关系结构对 $(\Delta\rho_k, \Delta\theta_k)$。以及每个关联所具备的容忍弹性系数 $(\lambda_\rho, \gamma_\theta)$。
- **模态从属性与完成标志**：GmemII 具备严格的模态信息。仅与 Anchor 节点模态相同的 GmemI 节点才允许被写入该节点。节点含 `is_completed` (布尔完成标记) 属性，用于状态流转时的挂起/恢复机制。
- **感性尺度 (Intrinsic Modality)**：不存储死板几何尺寸。改为记录探索用时计步 $T_{stay}$ 与极值跨长 $S_{span} = \max ||\mathbf{p}_i - \mathbf{p}_{init}||$，作为其内生的定性尺度。

### 4. GmemIII 宏观实体节点 (EntityNode)
宏观空间实体建构。每一次微观终结并发生大跨度跳跃转到新一轮物体端点前，所跨越的宏观眼跳向量 $\Delta\mathbf{p}$，会被作为空间位姿关系有向边连接到 GmemIII 模型中，实现“局部组件到宏观房屋”的认知拼装。

---

## 二、 动态投影与空间回放网络 (Gpos: Dynamic Projection Network)

Gpos 充当检索枢纽库。负责将孤立提取的特征张量 $\mathbf{w}$，放入不断变幻的输入视野流（多模矩阵字典 $X_{subspaces}$）中进行空间卷积找寻响度点。

### 1. 模版通道路由式投影 (L1: Modality Routing Projection)
仅当 GmemI 节点满足准入门限（处于 ACTIVE，并且没有在 REFRACTORY 态被隔离）且具有足月起充势量时：
将目标节点 $n$ 的核 $\mathbf{w}_n$ 对入所属通道源 $X_m$，并结合人工专家经验预置权重 $W_{mod}(m)$：
$$S_n(\mathbf{q}) = W_{mod}(m) \cdot \Big[ \mathbf{w}_n \ast X_{m}(\mathbf{q}) \Big]$$
（注：多通道特征投影响应匹配时，为压制杂波，可在通道余弦相似度夹角上施加非线性高阶幂 $k$：$S_m(\mathbf{q}) = ( \max ( 0, \frac{X_m \cdot w_m}{||X_m|| \cdot ||w_m|| + \epsilon} ) )^k$）

### 2. 拓扑同胚变换与结构共振网络 (L2: GposII Structure Synthesis)
面向在当前视野找回以往记忆拓扑网络（识别复现物体）。以锚为中心对所有挂外源启动一次强行同构核准。

**全息极坐标重采样 (Log-Polar Mesh Sampling)**
搜寻全图中最契合的锚点候补中心 $\mathbf{p}_{anc_{i}} = (x_c, y_c)$。在该点建立局部的对数极坐标网格变换：
$$r \leftarrow \log(\rho), \quad \theta \leftarrow \arctan\left(\frac{y}{x}\right)$$
把此时刻下所有从属周围节点产生的 L1 特征响应场 $S_k(\mathbf{q})$ 进行张量采样剥离扭曲，转化为以核心锚为参照坐标的高维特征圆环面响应图 $S_{LP}^{(k)}$。

**广义距离变换融合惩罚 (Generalized Distance Transform Approximation)**
各个外围节点根据记录记忆历史之中心位移理想点 $(\Delta \rho_k, \Delta \theta_k)$ 并结合配置好的弹性宽容系数度，利用平滑卷积内核执行粗化弥散逼近：
$$D^{(k)} = \mathcal{G}_{\text{blur}} \Big( S_{LP}^{(k)}(\rho - \Delta\rho_k, \theta - \Delta\theta_k) \Big)$$

**宏观结构共振坍缩 (Total Peak Resonance & Saccade Yield)**
汇聚出物体拓扑复燃结构图，其最高山峰指出此帧变形与偏差参数位姿：
$$D_{sum}(\rho_{grid}, \theta_{grid}) = \sum_{k} D^{(k)}(\rho_{grid}, \theta_{grid})$$
寻找 $\operatorname{argmax}(D_{sum})$，解出主体位置及扭转伸缩尺度（$\theta_{star}, s_{star}$）。

### 3. 宏观实体格式塔式检出 (L3: GposIII Gestalt Detection)
利用 GmemIII 中的 $(\Delta\rho, \Delta\theta)$ 位量边叠加与平滑中心聚类操作（如：$D_{sum}^{III}(\mathbf{q}) = \sum \mathcal{G}_{\text{blur}}(M_{GposII}(\mathbf{q} - \Delta\mathbf{p}))$），在相关局部组件集体点亮时涌现出一个总体的宏观概念峰值，支撑格式塔跨度识别。

---

## 三、 节点能量更新时机

在此算法架构下，图动力学演化严格划分为 **节点内源电位常驻时序引擎计算** 与 **网格拓扑更新操作** 两个相互独立的象限。为了保证代码架构解耦与物理意义上的准确，采用统筹式的独立节点更新范式。

### 1. 独立方法分封单步更新 (Independent Node Update Methods)
*   **机制分装**：将系统所有图结构节点的“能量电位更新”与“历时状态更新”进行脱钩与封装，归纳为独立的方法接口。
*   **单步尾随驱动**：在控制中心每一次执行 `run_step()` 结束帧，不论当前状态机正身处 `REVIEW`、`LEARN_MACRO` 或 `LEARN_MICRO` 哪一阶段，都将**统一进行一次全局属性调用**。传入需要受到时间轴推移影响或新受激充能的活跃节点集合，集中履行其自身的算力折损与能量结算算法。
*   **Gpos 响应联动同步**：在执行完节点的能量和状态更新后，配套执行对活跃状态的节点在 Gpos 中的空间响应谱系实时推演。保证下一帧的探索完全基于正确流变的物理力场。

### 2. 语义层级（GmemII）跨层捕获与向上传发激化
特设向上传递机制闭环，避免宏观防撞掩码模型失效引发局部死锁：
*   **操作时机 (MICRO 定向泵注)**：处于 `LEARN_MICRO` 时，每挖掘并挂载一个具有符合模态条件的子块环节后，除了点亮子要素的 GmemI 层外，必须利用独立更新向父级成型的语义中枢节点逆向输送受刺激能量参数 `E_input_gmem_ii_acc[active_semantic_id] += E_stimulus`，实现跨层直接激活。
*   **操作时机 (跨越连带激发)**：常规扫视时若恰切触碰高等级位元空间重合区，依然纳入独立结算列表内给予重算联觉能量的额外刺激。
*   **物理效果**：因独立结算机制保证了每一个时钟跳动下上代锚点电势均稳步攀登甚至红温顶格，一旦视野切换回宏观态（`LEARN_MACRO`），藉由该锚点散发的防撞辐射场 $M_{GposII}$ 或历史记录即可对眼跳重返造成绝对拒止。

---

## 四、 算法执行流与交互动力机制 (Execution Flow & Interaction Dynamics)

数据池 Gmem 等待激活，检索网 Gpos 提供空间映射引擎，而连接二者的核心动力即在于 **InterestOptimizer 空间动力学优化算子**与 **Controller 宏微观双轨眼跳伺服状态机**。两者的持续交火涌现出视觉注意力的自发转移路径。

### 1. 异构底层特征基场预处理 (Static Base Map Construction)
每输入一帧新图像体系，首先对其多通道进行属性解耦与过滤：
- **强度引力场** (STRENGTH): 纯振幅提取 $I_{str}(\mathbf{q}) = \sum w_{c} X_{str,c}(\mathbf{q})$
- **表面期望场** (CONTINUITY_SURFACE): 各向同性连续度平滑评估 $val_{surf}(\mathbf{q}) = \exp(-\frac{\|\nabla X_{surf,c}(\mathbf{q})\|^2}{2\sigma_{surf}^2})$
- **线域期望场** (CONTINUITY_TRACE): 求取局域局部张量场一致相干态，计算矢量模长的平滑衰减，聚合出 $I_{trace}(\mathbf{q})$。
融合构建出静态全视角起跳基盘：$I_{base}(\mathbf{q}) = I_{str}(\mathbf{q}) + w_s I_{surf}(\mathbf{q}) + w_t I_{trace}(\mathbf{q})$。

### 2. 空心化促进与绝压陨石坑双模空间调制场 (Spatial Activation Footprint)
眼跳扫描进程中，Gmem 记忆池中正在起振的节点将自身内生神经元振幅 $A_n(t)$ 投射回全图，演化为截然不同的动态阻滞空间足迹 $I_{spatial}$：
$$I_{spatial}^{(n)}(\mathbf{q}) = M_{resp}^{(n)}(\mathbf{q}) \cdot \Big( A_n(t) \cdot G_{local}(\mathbf{q}, \mathbf{p}_t) - |A_n(t)| \cdot G_{foveal}(\mathbf{q}, \mathbf{p}_t) \Big)$$
- **兴奋游走期 (ACTIVE, $A_n > 0$)**: 场表现为高斯差分（DoG 空心化促进场）。原心靶点处收益抵消趋 0，但在边外围产生诱导晕环，迫使系统自然向邻近未探索的同质介质滑步游走。
- **疲劳不应期 (REFRACTORY, $A_n = -1$)**: 在曾经走过的热区原地砸出极为深不可测的“连贯陨石坑”（Crater Inhibition）并下施外放全局阴影压制。

实时地形算谱将整合计算这片起伏不定的大陆拓扑：
$$I_{map}(\mathbf{q}) = w_{base} I_{base}(\mathbf{q}) + w_{spI} \sum I_{spatialI} + w_{spII} \sum I_{spatialII}$$

### 3. 微观纯化拓扑掩码的涌现 (Topologic Semantic Mask Diffusion)
在建立 GmemII 锚点 (Anchor) 初期，基于纯提取出来的 `CONTINUITY_SURFACE` 类型节点生成用于约束视野流的软拓扑掩码。
**通透率介质导通阻力 $P_{map}$**: 用纯面 Gpos 相似性 $M_{resp\_con}$，叠加 $I_{str}$ 获取之物理锐利物切边作流控阻墙阻断连通性：
$$P_{map}(\mathbf{q}) = M_{resp\_con}(\mathbf{q}) \cdot \operatorname{Sigmoid}(1 - \lambda \cdot I_{str}(\mathbf{q}))$$
**张量蔓延扩散引擎 (Max-Pool Tensor Dilation)**:
设定注视点原初星火 $V_0(\mathbf{p}_{macro}) = 1.0$。基于内置算子持续快速膨胀：
$$V_{t+1} = \text{MaxPool2d}(V_t) \odot P_{map}$$
直至完成预期探索直径内，截定锐化：$M_{semantic}(\mathbf{q}) = \operatorname{Sigmoid}(\gamma \cdot V_T(\mathbf{q}) - \delta)$

### 4. Controller 双极流状态机伺服循环 (Macro / Micro Saccade Control)

系统依凭底层场形动力构建一套全自动的扫描、探视再遗忘的高级流转。

#### 4.1 状态流转机制与条件
- **REVIEW / 初始 $\to$ LEARN_MACRO (全域猎捕跃迁)**: 当记忆巡视失去特定关注目标，或系统处于清空无拘束态（$M_{semantic} = null$），通过全局控制指令触发探索大循环。
- **LEARN_MACRO $\to$ LEARN_MICRO (锁区贪婪挖掘)**: 漫游寻找到未知的高显著极值落点，且未被目前涵盖全局建档防撞的历史掩码覆盖时。该点激发的事件自动升级为全新的基础特征锚点确立事件，即刻唤起拓扑掩码涌现，转入微观连续挖掘态。
- **LEARN_MICRO $\to$ LEARN_MACRO (辖区榨衰破溃 / 模态突变退出)**: 包含以下情况：
    1.  **常规耗尽退出**：每次眼跳不断蚕食微观约束图，导致当前局部的厌恶足迹覆盖率达标，或探测区域可用势能穷尽。高分节点看一个死一个，满坑满谷全是 REFRACTORY 负压陨石坑的掩模图内部全面崩坏坠崖。当全局探顶评估 $\max(I_{masked}) < \epsilon$ 时，确认此物体区域被解构收榨彻底完成，销毁软光圈掩码矩阵 $M_{semantic} \to \text{None}$。
    2.  **模态突变退出**：探测中视线逼近强边界导致强响应，诱发下一步注视点发生模态类型反转（$\text{Modality}(\mathbf{q}_{t+1}) \neq \text{Modality}(\mathbf{q}_t)$），强制触发流转退出。
    3.  **同源模态过滤与批量处理 (Batch Processing)**：在退出并转场瞬间，仅当前 `peripheral_buffer` 隔离区内与中心锚点节点模态信息相符的 GmemI 节点，才被允许批量推入并挂靠为该 GmemII 节点的子网络。
    4.  **未完成节点的挂起与恢复 (Suspend & Resume)**：若退出是由结构边界导致的模态突变（尚未全部探索完毕），系统会将该二维未完成 GmemII 节点的 `is_completed` 标记为 false 并压入栈中挂起，就地触发子级去专注首尾顺延新出现的异质特征群（如刚触碰的边缘）。当周边扰动特征群被学习闭环后，优先出栈重回该未完成的二维 GmemII 节点继续深耕，直至其区域探索枯竭将其 `is_completed` 置为 true。

#### 4.2 眼跳驱动策略

**1. REVIEW 状态（记忆验证阶段）**
*   **驱动目标**：对已有记忆特征（`interest_map`）高响应区域进行复查。
*   **计算模型**：注视点 $p_{next}$ 取决于记忆兴趣值与空间抑制的结合。
    $$p_{next} = \arg\max \left[ I_{map}(\mathbf{q}) - \alpha \cdot IoR(\mathbf{q}) - \gamma \cdot Dist(\mathbf{p}_{curr}, \mathbf{q}) \right]$$
*   **IoR演化**：全局抑制图 $IoR(\mathbf{q})$ 每次注视后在当前点 $\mathbf{p}_{curr}$ 进行高斯叠加并自然衰减，强制打破多次定焦死锁。

**2. LEARN_MACRO 状态（宏观锚点搜索阶段）**
*   **驱动目标**：寻找视觉中大面积连续平坦色块区域，并**严格规避已经探索建构的物体（语义特征团）辖区**。
*   **计算模型**：
    直接获取高阶位置图网络（GposII）对当前视野的真实空间位置响应图 $M_{GposII}$ 以及历史语义掩码作为绝对掩码约束，杜绝重返复建认知废墟。
    $$M_{suppress} = \max \left( \operatorname{Norm}(M_{GposII}) , c \cdot \operatorname{Norm} \left( \textstyle\sum M_{semantic\_history} \right) \right)$$
    $$I_{macro}(\mathbf{q}) = I_{con1}(\mathbf{q}) \odot \left(1 - M_{suppress} \right)$$
    利用形态学空间低通滤波 $K_{lowpass}$ 进行平滑以剔除高频孤立噪声：
    $$p_{next} = \arg\max \left( I_{macro} * K_{lowpass} \right)$$
    以此确保宏观观察点准确降落在未被建构认知区域的大面积连续面状色块的几何中心。

**3. LEARN_MICRO 状态（微观特征精细扫描阶段）**
*   **驱动目标**：在软掩模 $M_{semantic}$ 限制下的辖区内深度扫描，并实行与该视野面积特征自适应的眼跳约束。
*   **计算模型**：
    回归无偏的底层特征，且仅对软掩模辖区激发。
    $$p_{next} = \arg\max \left( S_{base}(\mathbf{q}) \odot M_{semantic}(\mathbf{q}) \odot (1 - M_{aversion}(\mathbf{q})) \right)$$
*   **动态扫描步长（自适应 $\sigma_{fovea}$）与厌恶足迹机制**：
    固定步长面对比例悬殊的掩码时，极易造成局部遍历过载或边缘越界。需引入几何面积向一维跨度投射的动态调节因子 $\sigma_{fovea}$：
    计算语义掩码面积积分 $A_{semantic} = \sum M_{semantic}(\mathbf{q})$
    进行量纲对齐并缩放生成步幅：
    $$\sigma_{fovea} = \eta \cdot \sqrt{A_{semantic}} + \epsilon_{base}$$
    每次产生新注视点 $\mathbf{p}_i$ 后，使用该自适应 $\sigma_{fovea}$ 在周围施加抑制分布：
    $$M_{aversion}^{(t+1)}(\mathbf{q}) = \max \left( M_{aversion}^{(t)}(\mathbf{q}) , \exp \left( -\frac{||\mathbf{q} - \mathbf{p}_i||^2}{2\sigma_{fovea}^2} \right) \right)$$
*   **退出条件计算**：每次眼跳检查自适应厌恶足迹对当前语义结构的覆盖率指标 $\rho$：
    $$\rho = \frac{\sum (M_{aversion} \odot M_{semantic})}{A_{semantic}}$$
    若 $\rho > \theta_{exit}$ （通常设为 $0.85$ 左右），表示该面状区内信息扫描已达期望水平，触发状态机无条件退出至 `LEARN_MACRO`。
*   **动态分辨率与细节挖掘**：
    在微观流中一旦多次捕捉到超高相似度且开始原地打转反复探测时，自适应调低 `similarity_threshold`，使得系统对特征更敏感，强行刺破表面大色块伪装，进行次级纹理精细切割。
