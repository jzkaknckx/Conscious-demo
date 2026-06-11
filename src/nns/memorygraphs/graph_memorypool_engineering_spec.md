

---

### 2. 宏观-微观双轨眼跳与语义掩码约束 (Macro/Micro Saccade & Semantic Mask)

针对特征跨越物体和漏挂关键点的问题，在 `LEARN` 阶段引入**大眼跳（Macro）与微眼跳（Micro）双态切换**，结合**动态语义掩码（Semantic Mask）**机制。此方案完美复用既有的 Gpos 与 Optimizer 动力学，仅在控制层增加“注意力光圈”。

#### 2.1 对原定掩码生成算法的缺陷评估与改进
**【原方案缺陷 - 漫水填充法 (Flood Fill) 的局限】**：
原设计尝试“利用 SWT 和 Hue 子空间执行一次漫水区域生长”。这种传统 CV 算法在深度学习特征场及复杂光照下极为硬直脆断。遇到渐变色、阴影切断会导致掩码过早截断；且在 Tensor 层面上执行图遍历开销极为巨大。

**【改进方案 - 基于 Gpos 核心的软掩码涌现 (Soft Emergence)】**：
系统无需外部引入漫水算法。当 Macro 眼跳降落并建锚时，系统本就会抽取当地的 CONTINUITY 特征送入 GmemI 活跃池。我们**直接利用这些 CONTINUITY 神经元向 Gpos 投递查询**，得出的空间响应热图 $M_{resp\_con}$ 便是一张自然的同质连通域权重。
将其叠加以起跳点为中心的高斯衰减，再经过 $\operatorname{Sigmoid}$ 锐化，即可自动“软涌现”出物体的拓扑掩码，完美平滑且可微。

#### 2.2 LEARN 阶段的新型双轨执行流设计

- **阶段 1：Macro 大跳降落与确立锚点**
  - **触发**：当前不存在 `active_semantic_id`。系统处于全图无掩码探索模式。
  - **动作**：
    1. 眼跳大步迈向全图最高点 $\mathbf{p}_{macro}$。建立 $GmemI$ 节点与 $GmemII$ 锚点（Anchor）。
    2. **生成语义软掩码 $M_{semantic}$**：借助上述的 Gpos 回响生成 $M_{semantic}$，确定“辖区光圈”。
    3. 状态自动锁定跨入 **Micro-Saccade**。依据掩码非零面积预估当前物体的尺度 $R_{scale}$。

- **阶段 2：Micro 微眼跳深耕与拓扑打包**
  - **触发**：持有活跃锚点且处于 Micro 态。
  - **动作**：
    1. **掩码约束全图兴趣**：每次 Optimizer 算出 $I_{map}$，强制抹除光圈外围噪音引诱：
       $$I_{masked} = I_{map} \odot M_{semantic}$$
    2. **自适应短步罚距**：眼跳受微观罚距 $\sigma_{micro}$（根据物体的 $R_{scale}$ 动态计算）限制，眼跳会死死咬住 $I_{masked}$ 掩码辖区内的未探索高点：
       $$\mathbf{p}_{t+1} = \operatorname{argmax} \left[ I_{masked} \cdot \exp\left(-\frac{\|\mathbf{q} - \mathbf{p}_t\|^2}{2\sigma_{micro}^2}\right) \right]$$
    3. **收割特征与砸出巨坑**：在掩码内获取特征形成 Peripheral 挂载到 Anchor，同时利用“满溢退出断层机制（带负权值 -1 的陨石坑）”，走过的地方被死死压平。

- **阶段 3：辖区破溃与爆发断层 (Breakout)**
  - **触发**：光圈内高光特征全部入账并跌入 REFRACTORY 不应期。导致局部引力场全面崩塌：
    $$\max(I_{masked}) < \epsilon$$
  - **动作**：
    清空该物体辖区掩码 $M_{semantic} \to \text{None}$，置空 `active_semantic_id`，解除当前锚的占有并封印记录。模型由 Micro 退回 Macro。重新朝向掩码外那片未被污染的新大陆进行大跨越。


# graph_memorypool.py 视知觉重构技术规格书 (V6.0)


---

## 2 Controller 基于宏观-微观双轨的掩码辖区注意力约束

针对特征跨越物体和漏掉边缘重点的情况，加入注意力光圈管辖，并在此时废弃单纯依靠空间同质性判定中断点。设计全新的 LEARN 三级执行流：

- **[修改点 2.1: 全局检索与 Macro-Saccade 降落]**
  - **触发**: 在无活跃 `SemanticNode` 锁定阶段。
  - **降落与起锚**: 利用不受限制的全局 $I_{map}$ 决定首跃落点 $\mathbf{p}_{\text{macro}}$。建立特征并创建出 $GmemII$ Anchor 记录基点。
  - **软掩码自动涌现**: 将起锚时录入的**所有 CONTINUITY 类通道**的神经元送回 Gpos 检索全图相似度 $M_{resp\_con}$，叠加局域高斯距离衰减限制后硬激活，从而直接得到软语义掩码（注意力光圈）：
    $$M_{semantic} = \operatorname{Sigmoid}(\alpha \cdot M_{resp\_con} \cdot \exp(-\frac{\|\mathbf{q} - \mathbf{p}_{macro}\|^2}{2\sigma_{mask}^2}) - \beta)$$
  - **换挡深耕**: 锁定当前 Anchor 为目标，依据掩码的实际非零积分面积，折算出物体范围 $R_{scale}$，立刻切换进入对该片区域的 Micro-Saccade 定制扫视态。

- **[修改点 2.2: 辖区内扫视深耕 (Micro-Saccade)]**
  - **光圈锁困**: 新一步 $I_{map}$ 获取后，强行叠加掩码以抹杀光圈外域信号：$I_{\text{masked}} = I_{map} \odot M_{semantic}$。
  - **防越境动态惩罚**: 眼跳惩罚函数不再使用全局游走的广阔长径。利用 $R_{scale}$ 收束一个狭小的罚径 $\sigma_{micro}$，确保眼动跳绳始终紧绑在当前对象的边界区域上。
  - **陨石坑埋扫收割**: 落点到未被踏查的局部强点上，建节点后挂载至 Anchor 的 `peripheral_links`。神经元进入活跃到不应期的衰退，抛出 Crater 坑式负引力压死当前区域。

- **[修改点 2.3: 辖区榨干与爆发退出 (Halt & Breakout)]**
  - **退出阈值**: 当微眼跳对光圈范围执行地毯式收割后，各个子块都变成绝压陨石坑。当判定全局残余收益极尽枯竭：
    $$\max(I_{\text{masked}}) < \epsilon$$
  - **清库流转**: 将当前 Anchor 标记归档为完成态。彻底抹去并销毁伴生的 $M_{semantic}$ 掩码张量，解除 `active_semantic_id` 锁定。重启 Macro-Saccade 去寻觅下一对象。
