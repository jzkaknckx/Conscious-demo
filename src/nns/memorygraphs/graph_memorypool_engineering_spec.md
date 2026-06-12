# Memory Graphs 理论框架重构与算法落地规范 (V6.0)

### 1. 连续性（CONTINUITY）属性的维数解耦与算子设计
在多通道特征解析中，由于图像天然的结构分布存在二元性（面与线段区别），"连续性"（CONTINUITY）必须在维度上被提取剥离，彻底解耦为**面连续性（CONTINUITY_SURFACE）**与**线连续性（CONTINUITY_TRACE）**。

#### 1.1 CONTINUITY_SURFACE (面域 / 二维域连续性)
- **物理意义**：代表大面积的颜色块、平缓的材质纹理（如 Hue 颜色通道或漫反射光照场）。在此类空间体上，连续意味着二阶导极为平缓，邻域表现出全向的同质性。
- **连续性评估算法 (用于生成底场与 GmemI 过滤门限)**：
  求解局部梯度幅值的均方作为空间差异度，通过负指数将其反转为局域平滑评估值：
  $$val_{surf}(\mathbf{p}) = \exp(-\frac{\|\nabla X_c(\mathbf{p})\|^2}{2\sigma_{surf}^2})$$
  该分值越高，说明该区域颜色/面状纹理越均匀纯净，是优良的表面描记锚点。

#### 1.2 CONTINUITY_TRACE (线域 / 一维迹连续性)
- **物理意义**：代表狭长、方向一致的轮廓、棱边或沟壑（如梯度角 Orientation、表面高斯曲率的主向）。这种特征的本能呈现为**仅沿切线方向连续**，而在法线方向上（哪怕是极小的位移距离）会发生剧烈的断层。若使用纯表面求导的 $\|\nabla X\|^2$，会将其错误地全部判定为不连续的杂乱噪声。
- **连续性评估算法 (用于生成底场与 GmemI 过滤门限)**：
  必须度量张量场（即局域内的单位方向向量 $\vec{v}(\mathbf{q})$）的**局部相干性一致度（Coherence）**。
  提取其向量加和项的标量长度与纯粹张量长度的加和比值，衡量其聚合态势：
  $$val_{trace}(\mathbf{p}) = \frac{\|\sum_{\mathbf{q} \in \Omega(\mathbf{p})} \vec{v}_c(\mathbf{q})\|}{\sum_{\mathbf{q} \in \Omega(\mathbf{p})} \|\vec{v}_c(\mathbf{q})\| + \epsilon}$$
  若局部区域内边缘走向一致（如一直线），各项分子不发生相消，连续性值极高（$\approx 1$）；若为杂乱的尖树叶团或无规律雪花点噪声，随机的法向互相抵消致使分子极小，连续性分值极低（$\approx 0$）。

---

### 2. 生成纯净语义掩码与宏/微观双轨眼跳流 (Macro/Micro Saccade)

在此前的架构中，掩码被错误地将 TRACE 型边缘等一并发起了连通域扩散和 Gpos 响唤，导致高分杂噪点与环境线索强行将掩码“粘连”到其它物体上，严重阻碍了 Saccade 判断。
因此必须切分出 **Macro 索敌** 和 **Micro 细嗅** 双态，并且**语义掩码 $M_{semantic}$ 仅限于 `CONTINUITY_SURFACE` 参与孕育**。

#### 2.1 基于 SURFACE 的语义光圈涌现 (Semantic Mask Generation)
当系统处于 **Macro-Saccade** 的自由探索状态，并在未开垦的新位置点 $\mathbf{p}_{macro}$ 第一次落点并挂建锚点对象 (Anchor) 时：
- **筛选提取**：系统从此时激活特征中，**严格过滤出仅归属于 `CONTINUITY_SURFACE` 属性的通道神经元**，坚决隔离任何 `STRENGTH` 强度或 `CONTINUITY_TRACE` 形貌线的入场！
- **Gpos 场回波**：将这些精纯的 SURFACE 节点送至 Gpos 检索全图，产生纯表面层级相似性空间响应热图 $M_{resp\_con}$。这屏蔽了环境中高频边界等尖锐物造成的虚假走廊。
- **掩码涌现计算**：
  将该面域回响加以着陆中心点的距离软惩罚并锐化推演，自动浮点映射出物体宽域身躯拓扑。
  $$M_{semantic}(\mathbf{q}) = \operatorname{Sigmoid}\left( \alpha \cdot \Big[ M_{resp\_con}(\mathbf{q}) \cdot \exp(-\frac{\|\mathbf{q} - \mathbf{p}_{macro}\|^2}{2\sigma_{mask}^2}) \Big] - \beta \right)$$

#### 2.2 LEARN 阶段的新型双轨执行流设计
基于这枚高质量软掩码，整体扫描生命流被赋予全新生命力：
1. **Macro 搜索大视态**：
   受全图（无掩码约束） $I_{map}$ 指引，寻找下一个物体落点 $\mathbf{p}_{macro}$。确立锚点，提取 SURFACE 类激活子计算并赋予掩码 $M_{semantic}$。锁定进入 Micro。
2. **Micro 局域辖区态 (光圈约束)**：
   - 核心兴趣受控：$I_{masked} = I_{map} \odot M_{semantic}$。
   - 自适应位流：受罚距引力 $\sigma_{micro}$（由掩码广度算出）管辖。在此阶段下，系统将在这个同色域身躯内部进行贪婪的拓扑收束，那些隐匿的面域间 `TRACE` 边缘高点、`STRENGTH` 特征峰会一一暴露在受限的 $I_{masked}$ 制高点被拾取挂载。
   - 当其眼跳落在高点并转不应期留下 Crater 负源底坑，其兴趣会平顺地逼向辖区里其他还未看过的同类结构边缘。
3. **退栈区决绝裂变 (Breakout)**：
   Micro 模式内部剩余最高预期 $\max(I_{masked}) < \epsilon$。标志着对象光圈里的结构全部被“看扁”。立即归正清空 $M_{semantic} \to \text{None}$，解放跳跃受迫锁重新回到 Macro。

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
  - **场回波重构**: 将余下的高纯面状介质（Hue团等）丢给 Gpos 泛提 $M_{resp\_con}$，叠加中心锚落点的相对距离衰减 $G_{dist\_decay}$ 作为调色，再整体进行 $\operatorname{Sigmoid}$。生成的高软隔离图作为新开出的 $M_{semantic}$ 掩码张量载入 Controller。
- **[修改点 2.2: Micro-Saccade 封锁态 (局部深耕)]**
  - **锁入机制**: 上个周期生成掩码及记录 `active_semantic_id`，立即锁转 Micro 引擎。
  - **场力斩截**: 生成出来的全图兴趣场必须通过 $M_{semantic}$ 阿尔法相乘：$I_{masked} = I_{map} \odot M_{semantic}$。从而绝育周边界限上的所有其余环境 $I_{base}$ 及引导力召唤。
  - **辖区罚距折算**: 将该软掩码非零域面积近似视作面积 $S$，推演出等效圆半径 $R_{scale}$，从而动态折返配用小粒度的视跨越跳行衰减距离 $\sigma_{micro} \propto R_{scale}$。使得跳步永远紧缩在本域。
- **[修改点 2.3: 陨坑满溢与退行破溃 (Breakout)]**
  - **判定阈值退行**: Micro 眼跳下经过的锚区将受到带有 $-1$ 不应权的空间双调制（大感受野正小极坑负）疯狂摧残，留下全是一路死寂的连串坑洞。当该区域被踏平使全区残值 $\max(I_{masked}) < \epsilon$ 时，标志当前物体已无法剥削。
  - **破笼重生**: 丢弃清空 $M_{semantic}$ 矩阵，$active\_semantic\_id \leftarrow \text{None}$，彻底解绑该 GmemII 组。放归为 Macro，跳眼被未踏足区域庞大的全新 Base Interest 远吸走。
