# Alg_optimizer_beta：视窗口向量引导优化器技术商讨

本文档讨论一条区别于旧版全局 `drive` 的优化器路线：在 MICRO 学习中，不再把整幅图上的兴趣场作为下一眼的直接目标，而是以当前注视点附近的视窗口为主要计算域，根据窗口内出现的特征支持维度、Gpos 实时响应、已观察覆盖与未闭合关系，合成一个下一眼移动向量。

该路线不否定 Gmem/Gpos 的基础设计。Gmem 仍保存节点与边，Gpos 仍负责把已学习节点和结构投影回感官场；改变的是优化器读取 Gpos 的方式：旧路线倾向于在全局响应图上选点，新路线倾向于在局部视窗口中估计运动方向。

---

## 1. 技术路线定位

### 1.1 旧路线：全局 drive 选点

旧优化器可抽象为：

$$
\mathbf{p}_{t+1}
=
\arg\max_{\mathbf{q}}
U_{global}(\mathbf{q})
$$

其中 $U_{global}$ 由 GposI/GposII 响应、新奇残差、访问抑制、距离项、期望项和基础显著性等全图矩阵加权得到。

其优点是：

1.  适合 MACRO 阶段在较大空间内发现新对象或新区域。
2.  适合 REVIEW 阶段用 GposII/GposIII 的结构响应找回已知对象。
3.  容易表达 top-down 期望，例如“已看到 A，因此去 A 的相对位置寻找 B”。

其缺点是：

1.  MICRO 阶段容易被全局最大值误导。大块二维 surface 的面积优势、强边缘的幅值优势、噪声残差的偶然峰值都可能支配 `argmax`。
2.  需要大量权重与抑制参数协调，否则视线会在连续特征内部打转，或在局部强边缘之间跳跃。
3.  对“沿一条线继续走”“沿一个面域外沿扩展”这类局部连续行为，整图最大值不是最自然的决策变量。

### 1.2 新路线：视窗口向量引导

新优化器将 MICRO 的下一眼写成：

$$
\mathbf{p}_{t+1}
=
\mathbf{p}_t
+
\operatorname{ClipStep}
\left(
\mathbf{v}_t,\,
s_{min},\,
s_{max}
\right)
$$

其中 $\mathbf{v}_t$ 只由当前视窗口 $\mathcal{W}_t$ 及其附近短时记忆计算：

$$
\mathcal{W}_t
=
\{\mathbf{q}\mid \|\mathbf{q}-\mathbf{p}_t\|\le r_w\}
$$

该路线的核心改变是：

1.  **从选点改为选方向**：优化器不问“整幅图哪里最高”，而问“当前看到的结构要求下一眼向哪里走”。
2.  **从全局竞争改为局部流形延展**：一维特征沿切向扩展，二维特征沿未覆盖外沿扩展，点事件作为局部高信息量吸引。
3.  **从机械状态切换改为动态竞争**：MACRO、MICRO、REVIEW 不再是固定流程，而是由局部学习价值、全局结构召回、未完成关系和探索疲劳共同竞争。

更合理的分工是：

| 模式 | 主要计算域 | 主要依据 | 行为 |
| --- | --- | --- | --- |
| MACRO | 全图或低分辨率全图 | 显著性、未知区域、远端 Gpos 响应 | 选择新的观察区域 |
| MICRO | 当前视窗口 | 局部 GposI、支持维度、frontier、工作关系边 | 连续学习当前 GmemII |
| REVIEW | 全图或候选结构区域 | GposII/GposIII 结构响应 | 验证、补全或修正已知结构 |

因此，新路线不是完全取消全局计算，而是把全局计算从 MICRO 的直接眼跳目标中移开。MICRO 用局部向量，MACRO 和 REVIEW 保留全局检索。

---

## 2. 每步实时 Gpos 响应

每一步都应实时计算 Gpos 响应，但不同层级的用途不同。

### 2.1 GposI：局部向量与窗口写入的基础

对活跃或候选 GmemI 节点 $n_i$：

$$
S_i^I(\mathbf{q})
=
\operatorname{Sim}
\left(
X_{m_i}(\mathbf{q}),\,
\mathbf{w}_i
\right)
\cdot
G_i(\mathbf{q})
\cdot
M_{valid}(\mathbf{q})
$$

GposI 至少在 $\mathcal{W}_t$ 内实时计算。若计算量允许，也可以对全图计算，但 MICRO 只读取局部窗口内的响应。

GposI 的用途包括：

1.  判断当前窗口内哪些位置属于已学习特征。
2.  判断哪些位置是低但非零的同类响应，用作下一采样点。
3.  判断哪些响应无法被现有节点解释，用于创建新 GmemI 节点。
4.  生成一维、二维、零维支持维度的局部统计量。

### 2.2 GposII/GposIII：状态竞争与 REVIEW 抢占

GposII/GposIII 的作用不是直接替代 MICRO 局部向量，而是作为状态竞争中的结构召回信号。若视窗口外某处出现强 GposII/GposIII 响应，说明当前输入中可能出现了已知结构，系统应优先 REVIEW。

设：

$$
R_{out}^{II/III}
=
\max_{\mathbf{q}\notin \mathcal{W}_t}
\left(
\max_h S_h^{II}(\mathbf{q}),\,
\max_e S_e^{III}(\mathbf{q})
\right)
$$

对应峰值位置为：

$$
\mathbf{q}_{review}^{*}
=
\arg\max_{\mathbf{q}\notin \mathcal{W}_t}
\left(
\max_h S_h^{II}(\mathbf{q}),\,
\max_e S_e^{III}(\mathbf{q})
\right)
$$

若 $R_{out}^{II/III}$ 超过抢占阈值，则当前 MACRO/MICRO 和正在学习的 GmemII 节点应挂起，系统立即进入 REVIEW：

$$
R_{out}^{II/III}>\theta_{review}^{preempt}
\quad
\land
\quad
\|\mathbf{q}_{review}^{*}-\mathbf{p}_t\|>r_w
$$

该机制符合“联想”目标：外部感官场中出现了与记忆结构强匹配的区域时，注意力不应继续机械完成当前局部学习，而应先验证这个已知结构是否真实存在。

---

## 3. 动态视窗口与尺度选择

视窗口向量路线必须引入动态尺度与动态形状。若窗口只是固定半径的圆，系统会在两个极端之间摇摆：小窗口能看到像素级细节，却难以把树、山脉、建筑轮廓作为整体；大窗口能看到整体，却会把叶脉、枝杈、纹理噪声和整体轮廓混在同一个写入流程中。

同时，真实对象的可学习区域通常不是圆形。更合理的做法是区分两类窗口：

$$
\mathcal{C}_t(r)
=
\{\mathbf{q}\mid \|\mathbf{q}-\mathbf{p}_t\|\le r\}
$$

$$
\mathcal{W}_t
=
\text{由 Gpos 响应域、当前 Gmem 节点和连通约束生成的工作视窗}
$$

其中 $\mathcal{C}_t(r)$ 只是计算上限、距离先验和注意力预算边界，真正参与写入和局部向量估计的是不规则工作视窗 $\mathcal{W}_t$。这使系统可以从局部细节起步，随着 GmemII 的填充和 Gpos 响应域扩大，逐渐把工作视窗扩展到覆盖整个物体或同一连续特征流形。

### 3.1 三种工作视窗方案的评判

**方案一：直接将 Gpos 响应的不规则区域作为工作视窗。**

设当前最相关响应为 $R_t^{work}(\mathbf{q})$，可直接取：

$$
\mathcal{W}_t
=
\operatorname{CC}_{seed}
\left(
\{\mathbf{q}\mid R_t^{work}(\mathbf{q})>\theta_w\}
\right)
$$

其中 $\operatorname{CC}_{seed}$ 表示包含当前注视点、当前 anchor 或最强 Gpos 峰值的连通分量。该方案最贴近对象真实形状，能自然得到不规则窗口；但它对 Gpos 噪声、断裂响应和多对象粘连很敏感。若直接使用全图 Gpos 连通域，视窗可能跨越远处相似纹理，破坏当前 GmemII 的局部性。

**方案二：将 Gpos 响应作为 mask 作用在圆形窗口上。**

$$
\mathcal{W}_t
=
\mathcal{C}_t(r_{cap})
\cap
\{\mathbf{q}\mid R_t^{work}(\mathbf{q})>\theta_w\}
\cap
M_{valid}
$$

该方案稳定、便宜，也便于继承原有半径与步长参数；但圆形边界仍会截断长条边缘、弯曲轮廓或大面积面域，使窗口形状受到人工几何先验限制。

**推荐方案：种子化 Gpos 响应域工作视窗。**

推荐保留半径或面积预算作为安全边界，但窗口形状由 Gpos 响应域决定：

$$
\mathcal{W}_t
=
\operatorname{Grow}_{seed}
\left(
R_t^{work},\,
B_t^{barrier},\,
A_t^{budget}
\right)
$$

其中 $R_t^{work}$ 是当前工作响应，$B_t^{barrier}$ 是阻断图，$A_t^{budget}$ 是注意力预算允许的最大有效面积。`Grow` 可以先用阈值和连通分量实现，后续再替换为地形代价扩散或测地距离扩张。

工作响应由当前语境决定：

$$
R_t^{work}(\mathbf{q})
=
\lambda_I
\max_{i\in\mathcal{A}_h}
S_i^I(\mathbf{q})
+
\lambda_{II}S_h^{II}(\mathbf{q})
+
\lambda_N N(\mathbf{q})
+
\lambda_F F_h(\mathbf{q})
$$

其中 $\mathcal{A}_h$ 是当前工作 GmemII 相关的 GmemI 节点集合。若尚未开启 GmemII，则用当前注视窗口内最强 GposI 原型或最稳定的新奇响应作为 seed：

$$
i^*
=
\arg\max_i
\max_{\mathbf{q}\in \mathcal{C}_t(r_{init})}
S_i^I(\mathbf{q})
$$

$$
R_t^{work}=S_{i^*}^I
\quad
\text{or}
\quad
R_t^{work}=N
$$

阻断图用于防止同色面域越过真实边界，也防止边缘追踪跨越无效区域：

$$
B_t^{barrier}
=
\lambda_g R_{grad}
+
\lambda_{tr} R_{RGB\_TRANSITION}
+
\lambda_{invalid}(1-M_{valid})
+
\lambda_{conflict}R_{modality\_conflict}
$$

对 surface 学习，$B_t^{barrier}$ 应抑制跨边界扩张；对 edge/line 学习，$B_t^{barrier}$ 不应抑制边缘本身，而应抑制法向逃逸和无效区域。

### 3.2 固定总注意力预算

用户提出的“视窗内总注意力为常数”可以形式化为一个预算约束。设总注意力预算为 $B_{att}$，不规则工作视窗的有效面积为：

$$
A_t
=
\sum_{\mathbf{q}}
\mathcal{W}_t(\mathbf{q})
$$

则单位面积注意力密度为：

$$
\beta_t
=
\frac{B_{att}}{A_t+\epsilon}
$$

当 $\mathcal{W}_t$ 覆盖面积增大时，$\beta_t$ 降低。其含义不是底层 CNN 看不到细节，而是优化器不允许把低预算下的细小响应写成独立节点；这些细节只能表现为纹理密度、粗糙度、边界复杂度或促使窗口收缩的尺度压力。

为兼容原有半径参数，可定义不规则窗口的等效半径：

$$
r_t^{eff}
=
\sqrt{\frac{A_t}{\pi}}
$$

对一个候选特征 $c_j$，设其支持维度为 $d_j$，等效空间尺度为 $\ell_j$，响应强度为 $R_j$。可定义尺度可写入能量：

$$
E_{write}(c_j;\mathcal{W}_t)
=
R_j
\cdot
\beta_t
\cdot
\left(
\frac{\ell_j}{\ell_0}
\right)^{d_j^{+}}
$$

其中 $d_j^{+}=\max(d_j,1)$，用于避免点事件的零维指数退化。写入条件为：

$$
E_{write}(c_j;\mathcal{W}_t)>\theta_{write,d_j}
$$

等价地，可以得到一个随窗口增大的最低写入尺度：

$$
\ell_{min,d}(\mathcal{W}_t)
=
\ell_{0,d}
\left(
\frac{A_t}{B_{att}}
\right)^{\eta_d}
$$

其中 $\eta_d$ 是支持维度相关的增长指数。工程上推荐使用更稳定的截断形式：

$$
\ell_{min,d}(\mathcal{W}_t)
=
\operatorname{Clip}
\left[
\ell_{base,d}
\left(
\frac{r_t^{eff}}{r_{min}}
\right)^{\eta_d},
\ell_{base,d},
\ell_{max,d}
\right]
$$

该式给出期望行为：

1.  小窗口时，$\ell_{min}$ 低，叶脉、枝杈、细边缘和像素级突变可以被写入。
2.  大窗口时，$\ell_{min}$ 高，小纹理不会成为独立 GmemI/GmemII 结构，只作为当前大结构的纹理统计或复杂度压力。
3.  若大量细节持续产生强残差，尺度竞争会推动 $\mathcal{W}_t$ 收缩到细节所在的 Gpos 响应域，进入细节观察。

### 3.3 候选工作域集合

为避免连续优化任意 mask，可使用对数间隔的候选面积预算：

$$
\mathcal{A}
=
\{A_1,A_2,\dots,A_K\}
$$

其中：

$$
A_k
=
\pi r_{min}^2
\left(
\frac{r_{max}^2}{r_{min}^2}
\right)^{\frac{k-1}{K-1}}
$$

对每个面积预算 $A_k$，从同一个 seed 出发生成一个候选工作域：

$$
\mathcal{W}_t^{(k)}
=
\operatorname{Grow}_{seed}
\left(
R_t^{work},\,
B_t^{barrier},\,
A_k
\right)
$$

最小实现可以使用以下迭代：

```text
W = seed
while area(W) < A_k:
    W_next = Dilate(W)
             ∩ [R_work > theta_low(k)]
             ∩ [B_barrier < theta_barrier]
             ∩ M_valid
    if W_next == W:
        break
    W = W_next
return W
```

然后选择：

$$
k^*
=
\arg\max_k Z_k(t)
$$

为了避免窗口尺度震荡，不直接替换 $\mathcal{W}_t$，而是平滑其面积或等效半径：

$$
\log r_{t+1}^{eff}
=
(1-\alpha_r)\log r_t^{eff}
+
\alpha_r\log r_{k^*}^{eff}
$$

同时限制单步面积变化：

$$
\left|
\log r_{t+1}^{eff}-\log r_t^{eff}
\right|
<
\Delta_{\log r}^{max}
$$

这样可以避免视窗口在叶脉尺度和整棵树尺度之间剧烈震荡，同时允许工作域边界沿 Gpos 响应的不规则形状生长。

### 3.4 尺度表征量

每个候选工作域 $\mathcal{W}^{(k)}$ 计算以下尺度表征：

| 表征量 | 定义 | 倾向 |
| --- | --- | --- |
| $G_k^{domain}$ | 候选工作域与 Gpos 响应域的连通一致性 | 选择贴合对象或连续特征的窗口 |
| $G_k^{scale}$ | GposI/GposII/GposIII 的响应尺度与 $r_k^{eff}$ 的匹配度 | 选择匹配已知结构或原型尺度的窗口 |
| $D_k^{fine}$ | 工作域内低于 $\ell_{min}(\mathcal{W}^{(k)})$ 的稳定细节残差 | 推动收缩窗口 |
| $C_k^{coarse}$ | 工作域内可被大尺度连贯解释的 surface/contour 强度 | 推动扩大或保持窗口 |
| $T_k^{frontier}$ | 当前结构 frontier 是否触及工作域外沿 | 推动扩张窗口 |
| $M_k^{mix}$ | 工作域内互不相容组件数量或模态冲突 | 推动收缩或分裂窗口 |
| $Q_k^{valid}$ | 有效视野占比与门控质量 | 排除 padding 和低质量窗口 |
| $H_k^{detail}$ | 细节熵或局部复杂度 | 在粗结构未稳定时推动收缩，在粗结构稳定后可作为纹理统计 |

#### 3.4.1 Gpos 响应域与响应尺度

对 GposI 响应图 $S_i^I$，可在阈值以上做连通域估计。二维 surface 的等效尺度为：

$$
s_i^{resp}
=
\sqrt{
\frac{
\sum_{\mathbf{q}}\mathbf{1}[S_i^I(\mathbf{q})>\theta_i]
}{\pi}
}
$$

一维 edge/line 的等效尺度可用骨架长度或主轴长度近似：

$$
s_i^{resp}
=
\lambda_{max}^{1/2}
\left(
\operatorname{Cov}
\{\mathbf{q}\mid S_i^I(\mathbf{q})>\theta_i\}
\right)
$$

对 GposII/GposIII，尺度来自关系边跨度：

$$
s_h^{II}
=
\max_{e\in h}
\|\Delta\mathbf{p}_e\|
$$

候选工作域与响应尺度的匹配度为：

$$
G_k^{scale}
=
\max_j
S_j^{peak}
\exp
\left(
-\frac{
(\log r_k^{eff}-\log(\kappa_s s_j))^2
}{2\sigma_s^2}
\right)
$$

其中 $s_j$ 可来自 GposI 响应域或 GposII/GposIII 结构跨度，$\kappa_s$ 表示窗口应略大于目标结构。

响应域一致性可用软 IoU 表示：

$$
G_k^{domain}
=
\frac{
\sum_{\mathbf{q}}
\mathcal{W}^{(k)}(\mathbf{q})
R_t^{work}(\mathbf{q})
}{
\sum_{\mathbf{q}}
\max
\left(
\mathcal{W}^{(k)}(\mathbf{q}),
R_t^{work}(\mathbf{q})
\right)
+\epsilon
}
$$

若 $G_k^{domain}$ 低，说明候选窗口主要由圆形上限或扩张操作产生，而不是由当前特征响应支撑，不应继续扩大。

#### 3.4.2 细节残差与粗结构连贯度

设工作域内候选特征集合为 $\mathcal{C}(\mathcal{W}^{(k)})$。低于当前最低写入尺度的稳定响应不直接写入，而累计为细节压力：

$$
D_k^{fine}
=
\sum_{c_j\in\mathcal{C}(\mathcal{W}^{(k)})}
\mathbf{1}[\ell_j<\ell_{min,d_j}(\mathcal{W}^{(k)})]
R_j
\left(
1-E_j
\right)
$$

其中 $E_j$ 是现有 Gmem/Gpos 对该候选的解释度。

粗结构连贯度可用大尺度 surface/contour 的可解释覆盖表示：

$$
C_k^{coarse}
=
\frac{
\sum_{\mathbf{q}}
\mathcal{W}^{(k)}(\mathbf{q})
E_{coarse}(\mathbf{q})
}{
\sum_{\mathbf{q}}
\mathcal{W}^{(k)}(\mathbf{q})
R_{coarse}(\mathbf{q})+\epsilon
}
$$

当 $C_k^{coarse}$ 高时，说明大窗口不是把无关细节混在一起，而是在观察一个可被整体解释的粗结构。

### 3.5 工作域竞争函数

候选工作域的分数可定义为：

$$
Z_k
=
w_{domain}G_k^{domain}
+
w_g G_k^{scale}
+
w_c C_k^{coarse}
+
w_f T_k^{frontier}
+
w_q Q_k^{valid}
-
w_{fine}D_k^{fine}
-
w_m M_k^{mix}
-
w_s
\left|
\log r_k^{eff}-\log r_t^{eff}
\right|
$$

其中 $D_k^{fine}$ 不应永远惩罚大窗口。推荐加入粗结构完成度调制：

$$
w_{fine}
=
w_{fine}^0
\left(
1-C_h^{coarse}
\right)
$$

含义是：当粗结构尚未建立时，细节残差更应该促使窗口缩小；当粗结构已经稳定时，细节残差可以被记录为纹理统计，只有在任务需要或残差长期稳定时才触发细节观察。

若要更明确地表达“先整体，后细节”的人类式观察，可增加粗到细的预算门：

$$
\psi_{detail}
=
\sigma
\left(
\frac{C_h^{coarse}-\theta_{coarse}}{\tau_c}
\right)
$$

当 $\psi_{detail}$ 低时，系统偏向寻找整体轮廓；当 $\psi_{detail}$ 高且 $D_k^{fine}$ 高时，系统允许缩小窗口学习细节。

### 3.6 动态窗口对特征写入的约束

写入特征时，候选点必须满足尺度门控：

$$
\ell_j\ge \ell_{min,d_j}(\mathcal{W}_t)
$$

或满足强事件例外：

$$
R_j>\theta_{event}^{strong}(\mathcal{W}_t)
$$

其中强事件阈值随窗口增大而提高：

$$
\theta_{event}^{strong}(\mathcal{W}_t)
=
\theta_{event}^{0}
\left(
\frac{r_t^{eff}}{r_{min}}
\right)^{\eta_0}
$$

因此，大窗口下只有山脊轮廓、树冠外沿、建筑边界等大尺度稳定结构会被写入；叶脉、树叶边缘、纹理点不会被静默丢弃，而是进入以下三类统计：

1.  `texture_density`：局部细节总量。
2.  `detail_entropy`：细节方向、颜色或频率分布复杂度。
3.  `detail_pressure`：未来是否需要缩小窗口的残差信号。

当系统进入小窗口时，这些细节才被允许成为独立 GmemI 节点和 GmemII 关系。

### 3.7 动态窗口对眼动向量的影响

工作视窗改变的不只是可见范围，也改变动态矩阵和向量合成的物理尺度：

$$
s_{min}(\mathcal{W}_t)=\kappa_{min}r_t^{eff},\quad
s_{max}(\mathcal{W}_t)=\kappa_{max}r_t^{eff}
$$

边缘蝴蝶核的预测步长应随窗口变化：

$$
s_e(\mathcal{W}_t)=\kappa_e r_t^{eff}
$$

surface frontier 的膨胀半径也应随窗口变化：

$$
\delta_{surf}(\mathcal{W}_t)=\kappa_{surf}r_t^{eff}
$$

访问抑制同样按尺度分层：

$$
O_t
=
O_t^{fine}
\oplus
O_t^{mid}
\oplus
O_t^{coarse}
$$

小窗口主要更新 $O^{fine}$，大窗口主要更新 $O^{coarse}$。这样模型在看完整棵树的外轮廓后，不会因为叶片级访问抑制而错误认为整体轮廓已经覆盖；反过来，观察叶脉细节时，也不会破坏粗尺度树冠 frontier。

### 3.8 最小可实现流程

动态视窗口的最小流程为：

```text
Step A: 构造工作响应
    若存在 h_work:
        R_work = active GmemI/GposI + GposII + frontier + novelty
    否则:
        R_work = 当前局部最强 GposI 或稳定新奇残差

Step B: 生成候选工作域
    A = logspace(pi*r_min^2, pi*r_max^2, K)
    对每个 A_k:
        W_k = seeded_grow(R_work, barrier, A_k)
        估计 Gpos 响应域一致性 G_k^{domain}
        估计 Gpos 响应尺度匹配 G_k^{scale}
        估计细节残差 D_k^{fine}
        估计粗结构连贯度 C_k^{coarse}
        估计 frontier 触边 T_k^{frontier}
        估计多对象混合 M_k^{mix}
        计算 Z_k

Step C: 平滑选择工作域
    k* = argmax Z_k
    W_t = smooth_or_hysteresis(W_{t-1}, W_{k*})
    r_eff = sqrt(area(W_t)/pi)

Step D: 按尺度执行窗口写入
    只写入 size >= ell_min_d(W_t) 的关键点
    小于阈值的稳定细节写入纹理统计和 detail_pressure

Step E: 按尺度执行局部向量
    使用 s_min(W_t), s_max(W_t), s_e(W_t), delta_surf(W_t)
    合成下一眼向量
```

该流程只需要少量候选面积预算上的阈值、连通扩张和池化，不要求复杂矩阵优化，也不要求引入传统 attention。它把“整体轮廓”和“细节纹理”的切换归结为同一件事：在固定注意力预算下选择当前最值得解释的 Gpos 响应域。

---

## 4. 基于动态竞争的状态机

新路线仍保留 MACRO、MICRO、REVIEW 三种行为模式，但它们不再由固定 if-else 流程切换，而是由动态分数竞争。状态是行为吸引子，不是手写任务脚本。

### 4.1 状态集合

推荐状态集合为：

| 状态 | 含义 |
| --- | --- |
| `MACRO_SEARCH` | 全局或低分辨率寻找下一观察区域 |
| `MICRO_BUILD` | 以视窗口向量补全当前 GmemII |
| `REVIEW_VERIFY` | 跳向 GposII/GposIII 强响应位置验证已知结构 |
| `SUSPEND_RESUME` | 从 REVIEW 或失败恢复后回到被挂起的 GmemII |

`SUSPEND_RESUME` 可以是显式状态，也可以是状态切换前后的栈操作。工程上建议先做成显式状态，便于调试。

### 4.2 动态观测量

每一步计算以下低维观测量：

| 观测量 | 定义 | 作用 |
| --- | --- | --- |
| $N_W$ | 视窗口内未解释稳定响应 | 支持 MICRO 新学习 |
| $F_W$ | 当前 GmemII 的局部 frontier 强度 | 支持 MICRO 延展 |
| $Q_W$ | 当前窗口内特征门控质量 | 抑制噪声窗口 |
| $C_h$ | 当前 GmemII 完成度 | 完成度高则降低 MICRO |
| $R_{out}^{II/III}$ | 视窗口外结构召回峰值 | 支持 REVIEW 抢占 |
| $R_{in}^{II/III}$ | 视窗口内结构匹配峰值 | 支持 REVIEW 确认或恢复 |
| $G_h$ | 当前 GmemII 未闭合关系缺口 | 支持继续补全 |
| $E_{fail}$ | 局部向量失败或震荡次数 | 支持 MACRO 或挂起 |
| $A_{state}$ | 当前状态持续时间与惯性 | 提供迟滞 |

其中：

$$
N_W
=
\sum_{\mathbf{q}\in\mathcal{W}_t}
R_{raw}(\mathbf{q})
\left(
1-\max_i S_i^I(\mathbf{q})
\right)
$$

$$
F_W
=
\sum_{\mathbf{q}\in\mathcal{W}_t}
F_h(\mathbf{q})
$$

$$
C_h
=
w_o C_{obs}
+
w_e C_{edge}
+
w_r C_{relation}
-
w_n N_W
$$

这里 $F_h$ 是当前正在生长的 GmemII 的 frontier 图，$C_h$ 是完成度估计。

### 4.3 状态分数

每个状态都有一个竞争分数：

$$
z_s(t)
=
b_s
+
\sum_j
w_{s,j}\phi_j(t)
+
\rho_s \mathbf{1}[s=s_t]
-
\pi_s(t)
$$

其中 $\phi_j(t)$ 是上节观测量，$\rho_s$ 是状态惯性，$\pi_s$ 是失败惩罚或资源惩罚。

可使用 softmax 得到状态概率：

$$
P(s\mid t)
=
\frac{\exp(z_s/\tau)}
{\sum_{s'}\exp(z_{s'}/\tau)}
$$

工程上不必随机采样，直接采用带迟滞的 winner-take-all：

$$
s_{t+1}
=
\begin{cases}
s^*, & z_{s^*}>z_{s_t}+\Delta_{switch}
\land age(s_t)>T_{min}\\
s_t, & \text{otherwise}
\end{cases}
$$

其中：

$$
s^*=\arg\max_s z_s
$$

### 4.4 推荐分数形式

MICRO 分数：

$$
z_{micro}
=
b_{micro}
+
a_1 N_W
+
a_2 F_W
+
a_3 G_h
+
a_4 Q_W
-
a_5 C_h
-
a_6 E_{fail}
+
\rho_{micro}\mathbf{1}[s_t=MICRO]
$$

MACRO 分数：

$$
z_{macro}
=
b_{macro}
+
m_1 E_{fail}
+
m_2 (1-Q_W)
+
m_3 C_h
+
m_4 U_{global}^{novel}
-
m_5 R_{out}^{II/III}
+
\rho_{macro}\mathbf{1}[s_t=MACRO]
$$

REVIEW 分数：

$$
z_{review}
=
b_{review}
+
r_1 R_{out}^{II/III}
+
r_2 R_{in}^{II/III}
+
r_3 G_{topdown}
-
r_4 E_{review\_recent}
+
\rho_{review}\mathbf{1}[s_t=REVIEW]
$$

其中 $G_{topdown}$ 表示已有激活结构对缺失关系的预测强度，$E_{review\_recent}$ 抑制刚刚失败或刚刚验证过的结构重复抢占。

### 4.5 REVIEW 抢占与挂起

REVIEW 抢占应高于普通迟滞规则：

$$
R_{out}^{II/III}>\theta_{review}^{preempt}
\quad
\land
\quad
z_{review}>\max(z_{micro},z_{macro})+\Delta_{preempt}
$$

若满足条件：

1.  将当前上下文压入 `suspend_stack`：
    $$
    ctx_t=(state,\ sem\_id,\ \mathbf{p}_t,\ F_h,\ O_h,\ e^{work},\ age,\ confidence)
    $$
2.  将 REVIEW 目标设为 $\mathbf{q}_{review}^{*}$。
3.  执行大跨度眼跳到 REVIEW 目标附近。
4.  REVIEW 成功时，更新对应 GmemII/GmemIII 的置信度和关系边。
5.  REVIEW 失败时，降低该结构的短时抢占权重；若局部出现高新奇响应，可在当前位置开启新的 MICRO。
6.  REVIEW 结束后，若 `suspend_stack` 中仍有未完成 GmemII 且其恢复价值高，则进入 `SUSPEND_RESUME`，否则回到 `MACRO_SEARCH`。

恢复价值可定义为：

$$
V_{resume}
=
\eta_1(1-C_h)
+
\eta_2 F_h^{remain}
+
\eta_3 G_h
-
\eta_4 age_{suspend}
$$

若 $V_{resume}>\theta_{resume}$，回到被挂起的 GmemII；否则将其保留为未完成节点，等待未来被 GposII 或 MACRO 再次唤醒。

---

## 5. 动态引导矩阵与局部眼动向量

### 5.1 动态矩阵定义

维护一个与当前视野同尺寸的短时动态矩阵：

$$
P_t\in\mathbb{R}^{H\times W}
$$

它不是全局 drive。它不直接通过整图 $\arg\max$ 决定下一眼，而是作为局部向量场的势能记忆。每一步只在视窗口及其邻域更新：

$$
P_{t+1}
=
\operatorname{Clip}
\left(
\lambda_P P_t
+
U_t^{edge}
+
U_t^{surface}
+
U_t^{event}
+
U_t^{gap}
-
U_t^{return},
0,
1
\right)
$$

其中 $\lambda_P$ 是短时衰减，$U_t^{return}$ 来自位置级、节点级和关系边级访问抑制。

下一眼向量由局部加权一阶矩给出。由于 $\mathcal{W}_t$ 已是不规则工作域，候选集合不再是简单圆环，而是工作域外沿、工作域邻域和步长约束的交集：

$$
\Omega_t^{vec}
=
\operatorname{Dilate}
\left(
\mathcal{W}_t,\delta_{out}
\right)
\cap M_{valid}
$$

$$
\partial\mathcal{W}_t
=
\operatorname{Dilate}
\left(
\mathcal{W}_t,\delta_b
\right)
-
\operatorname{Erode}
\left(
\mathcal{W}_t,\delta_b
\right)
$$

$$
\mathcal{A}_t
=
\Omega_t^{vec}
\cap
\left[
\partial\mathcal{W}_t
\cup
\{\mathbf{q}\mid s_{min}(\mathcal{W}_t)\le \|\mathbf{q}-\mathbf{p}_t\|\le s_{max}(\mathcal{W}_t)\}
\right]
$$

则：

$$
\mathbf{v}_{P}
=
\frac{
\sum_{\mathbf{q}\in\mathcal{A}_t}
P_t(\mathbf{q})
(\mathbf{q}-\mathbf{p}_t)
}{
\sum_{\mathbf{q}\in\mathcal{A}_t}
P_t(\mathbf{q})+\epsilon
}
$$

该式只取局部方向，不取全图最高点。

### 5.2 一维边缘的蝴蝶状响应

对窗口内边缘点 $\mathbf{x}$，设边缘强度为 $E(\mathbf{x})$，边缘切向为 $\hat{\mathbf{t}}(\mathbf{x})$。若输入的 `ori` 表示线轴方向，则：

$$
\hat{\mathbf{t}}=(\cos ori,\ \sin ori)
$$

若输入的 `grad` 方向表示法向，则应旋转 $90^\circ$ 得到切向。

定义位移：

$$
\mathbf{r}=\mathbf{q}-\mathbf{x}
$$

蝴蝶状核：

$$
K_{butterfly}
(\mathbf{r},\hat{\mathbf{t}})
=
\exp
\left(
-\frac{(\|\mathbf{r}\|-s_e)^2}{2\sigma_r^2}
\right)
\left[
\exp
\left(
-\frac{\angle(\mathbf{r},\hat{\mathbf{t}})^2}{2\sigma_\theta^2}
\right)
+
\exp
\left(
-\frac{\angle(\mathbf{r},-\hat{\mathbf{t}})^2}{2\sigma_\theta^2}
\right)
\right]
$$

该核沿 $\hat{\mathbf{t}}$ 与 $\hat{\mathbf{t}}+\pi$ 两个方向最强，沿法向最弱，形成两瓣结构。边缘对动态矩阵的写入为：

$$
U_t^{edge}(\mathbf{q})
=
\sum_{\mathbf{x}\in\mathcal{W}_t}
E(\mathbf{x})
K_{butterfly}
(\mathbf{q}-\mathbf{x},\hat{\mathbf{t}}(\mathbf{x}))
\left(
1-O_{edge}(\mathbf{q})
\right)
M_{valid}(\mathbf{q})
$$

若一段连续边缘已经被观察，内部点的前后蝴蝶瓣会落在已观察轨迹上，并被 $O_{edge}$ 抑制；只有连续轨迹两端之外的位置仍保留高响应。因此在理想情况下，动态矩阵会自然在已观察边缘的两个外端形成最高响应。

为了避免边缘被错误延展，真正选向量时还应乘以 GposI 的低阈值支持：

$$
L_{edge}(\mathbf{q})
=
\mathbf{1}
\left[
\theta_{edge}^{low}
<
S_{edge}^{I}(\mathbf{q})
<
\theta_{edge}^{high}
\right]
$$

最终一维候选势能为：

$$
P_t^{edge}
=
U_t^{edge}
\odot
\left(
\lambda_c \mathbf{1}[S_{edge}^{I}>\theta_{edge}^{high}]
+
\lambda_l L_{edge}
\right)
$$

其中 $L_{edge}$ 对应“低但非零”的同类响应，用于引导下一采样点，而高响应区域主要用于确认和更新已知边缘。

### 5.3 二维 surface 的外沿响应

二维连续特征不能直接使用普通高斯吸引。若对已观察 surface 点简单叠加正高斯：

$$
U(\mathbf{q})
=
\sum_{\mathbf{x}\in O_{h,i}^{surf}}
G_\sigma(\mathbf{q}-\mathbf{x})
$$

最高响应会落在已观察内部，容易诱导回看。二维 surface 需要的是“外延壳层”而不是“中心吸引”。因此，高斯可以使用，但应以墨西哥帽、差分高斯或锥形距离核的形式出现，并且必须乘以同类 Gpos 响应与边界阻断。

对某个 surface 原型 $i$，令软响应域为：

$$
M_i^{surf}(\mathbf{q})
=
\operatorname{Clip}
\left(
\frac{S_i^I(\mathbf{q})-\theta_{surf}^{low}}
{\theta_{surf}^{high}-\theta_{surf}^{low}+\epsilon},
0,
1
\right)
$$

令 $O_{h,i}^{surf}$ 是当前 GmemII 已观察到的该 surface 覆盖。推荐的外沿壳层为：

$$
F_{h,i}^{shell}
=
\left[
\operatorname{DoG}^{+}_{\sigma_o,\sigma_i}
*
O_{h,i}^{surf}
\right]
\odot
M_i^{surf}
\odot
\left(
1-O_{h,i}^{surf}
\right)
\odot
\left(
1-B_t^{barrier}
\right)
$$

其中：

$$
\operatorname{DoG}^{+}_{\sigma_o,\sigma_i}
=
\operatorname{ReLU}
\left(
G_{\sigma_o}
-
\lambda_i G_{\sigma_i}
\right),
\quad
\sigma_o>\sigma_i
$$

该核在已观察区域附近形成正壳层，在内部被 $\left(1-O_{h,i}^{surf}\right)$ 消除，因此势能会落在 surface 外沿，而不是落在内部。

若希望更便宜，可用锥形距离核替代：

$$
D_i^{out}(\mathbf{q})
=
\operatorname{dist}
\left(
\mathbf{q},O_{h,i}^{surf}
\right)
$$

$$
F_{h,i}^{cone}
=
\max
\left(
0,
1-\frac{D_i^{out}}{\delta_{surf}}
\right)
\odot
M_i^{surf}
\odot
\left(
1-O_{h,i}^{surf}
\right)
\odot
\left(
1-B_t^{barrier}
\right)
$$

最终 surface 势能为：

$$
U_t^{surface}
=
\sum_i
\omega_i
\left[
\lambda_{shell}F_{h,i}^{shell}
+
\lambda_{cone}F_{h,i}^{cone}
+
\lambda_{low}L_i^{surf}
\right]
\odot
\left(
1-O_{node,i}^{wide}
\right)
$$

其中：

$$
L_i^{surf}(\mathbf{q})
=
\mathbf{1}
\left[
\theta_{surf}^{low}
<
S_i^I(\mathbf{q})
<
\theta_{surf}^{high}
\right]
$$

`L_i^{surf}` 对应“同类 surface 的低但非零响应”，常位于颜色过渡、阴影、模糊边界或尺度变化处。它不单独决定写入，只作为外沿采样的辅助。

二维 surface 的延伸方向不是一维边缘的两个方向，而是一组连续角度。简单向量可由外沿壳层的一阶矩得到：

$$
\mathbf{v}_{surf,i}
=
\frac{
\sum_{\mathbf{q}\in\Omega_t^{vec}}
F_{h,i}^{shell}(\mathbf{q})
(\mathbf{q}-\mathbf{p}_t)
}{
\sum_{\mathbf{q}\in\Omega_t^{vec}}
F_{h,i}^{shell}(\mathbf{q})+\epsilon
}
$$

若 surface 外沿在多个方向都很强，优化器不应强行压缩成一个方向；可以保留多个局部峰，按注意力预算逐个采样，或交给动态工作视窗在后续步骤中继续扩张。该机制的目标不是完整填满面域，而是用少量样本估计内部颜色/纹理统计，并把注意力推向面域外沿，以找到边界、孔洞、相邻对象或过渡带。

### 5.4 点事件、关系缺口与其他因素

零维事件包括角点、端点、交点、终止点和局部突变。它们不需要复杂延展，只需在局部峰值处提高势能：

$$
U_t^{event}(\mathbf{q})
=
\sum_k
\omega_k
S_k^{I}(\mathbf{q})
\mathbf{1}
\left[
S_k^I(\mathbf{q})>\theta_k
\right]
\left(
1-O_k^{point}(\mathbf{q})
\right)
$$

未闭合关系边产生缺口向量。设工作关系边预测下一点为 $\hat{\mathbf{q}}_e$：

$$
U_t^{gap}(\mathbf{q})
=
\sum_e
(1-confidence_e)
\exp
\left(
-\frac{
\|\mathbf{q}-\hat{\mathbf{q}}_e\|^2
}{2\sigma_e^2}
\right)
S_{compatible}^{I}(\mathbf{q})
$$

其中 $S_{compatible}^{I}$ 是与该关系边支持维度和模态相容的 GposI 响应。

### 5.5 多因素合成眼动向量

最终眼动向量不是单一矩阵的梯度，而是多个局部向量的叠加：

$$
\mathbf{v}_t
=
\lambda_P\mathbf{v}_P
+
\lambda_{low}\mathbf{v}_{lowGpos}
+
\lambda_{gap}\mathbf{v}_{gap}
+
\lambda_{nov}\mathbf{v}_{novel}
+
\lambda_{inertia}\mathbf{v}_{t-1}
-
\lambda_{return}\mathbf{v}_{return}
$$

其中：

*   $\mathbf{v}_P$ 来自动态矩阵 $P_t$ 的局部一阶矩。
*   $\mathbf{v}_{lowGpos}$ 指向 GposI 响应低但非零的同类区域，用于采样边缘或 surface 的外延变化。
*   $\mathbf{v}_{gap}$ 指向未闭合关系边的预测位置。
*   $\mathbf{v}_{novel}$ 指向窗口内稳定但未解释的新奇响应。
*   $\mathbf{v}_{t-1}$ 提供短时惯性，尤其用于一维追踪。
*   $\mathbf{v}_{return}$ 远离刚刚访问或已经闭合的片段。

若 $\|\mathbf{v}_t\|$ 过小，说明局部信息对称或缺乏可延展线索。此时不应随机抖动，而应增加 $E_{fail}$，让状态竞争机制转向 MACRO 或 REVIEW。

---

## 6. 视窗口下的特征写入模式

新路线要求每次注视不只写入中心点特征，而是写入视窗口内的关键点集合。关键点集合既服务于 GmemI 原型学习，也服务于 GmemII 关系边学习。

动态视窗口机制加入后，关键点写入还必须受当前工作域 $\mathcal{W}_t$ 的有效面积控制。工作域越大，允许写入的最低特征尺度越大；工作域越小，才允许细节成为独立节点。

### 6.1 视窗口特征抽取

每次注视取窗口：

$$
X_t^W
=
X_t[\mathcal{W}_t]
$$

在窗口内计算：

1.  原始特征门控 $R_m(\mathbf{q})$。
2.  活跃 GmemI 的 GposI 响应 $S_i^I(\mathbf{q})$。
3.  已解释度：
    $$
    E(\mathbf{q})
    =
    \max_i S_i^I(\mathbf{q})
    $$
4.  新奇残差：
    $$
    N(\mathbf{q})
    =
    R_{raw}(\mathbf{q})(1-E(\mathbf{q}))
    $$
5.  支持维度统计：
    $$
    M_d
    =
    \sum_{\mathbf{q}\in\mathcal{W}_t}
    \sum_{i:support\_dim_i=d}
    S_i^I(\mathbf{q})
    $$

### 6.2 关键点采样原则

为避免复杂矩阵运算，关键点可采用阈值加局部极值抑制：

1.  对每个支持维度 $d\in\{0,1,2\}$ 生成候选图。
2.  对候选图做小窗口 max pooling 或简单非极大值抑制。
3.  每个支持维度最多取 $K_d$ 个点。
4.  保证关键点之间距离大于 $r_{nms}$。
5.  丢弃 $\ell_j<\ell_{min,d_j}(\mathcal{W}_t)$ 的普通候选，只把它们累计为纹理统计或细节压力。
6.  优先保留离当前注视点不太近、且不在已访问覆盖中的点。

关键点分为四类：

| 类型 | 候选条件 | 写入目的 |
| --- | --- | --- |
| `known_confirm` | $S_i^I>\theta_i^{high}$ | 更新已有 GmemI 统计与 relation 置信度 |
| `known_extend` | $\theta_i^{low}<S_i^I<\theta_i^{high}$ | 采样同类特征的边缘、端点或外延变化 |
| `novel_seed` | $N>\theta_{novel}$ 且门控稳定 | 创建或暂存新 GmemI |
| `event_peak` | 点事件响应超过阈值 | 写入角点、端点、交点等结构事件 |

其中 `known_extend` 是这条路线的关键。重复的一维或二维特征在视窗口内出现时，不应继续采样最高 Gpos 响应点，而应优先选择 Gpos 响应低但非零的位置：

$$
\theta_i^{low}<S_i^I(\mathbf{q})<\theta_i^{high}
$$

这些位置往往位于响应流形的边缘、端点、过渡区或尺度变化处，信息量高于内部峰值。

### 6.3 GmemI 写入与复用

对每个关键点 $\mathbf{k}_j$，提取局部描述符：

$$
d_j
=
\left(
modality,\ values,\ sim\_mask,\ gate\_context,\ topology\_context
\right)
$$

写入规则：

1.  若存在相似 GmemI 节点 $n_i$ 且相似度超过阈值，则更新该节点原型与统计。
2.  若不存在相似节点，但门控强、响应稳定，则创建新的 GmemI 节点。
3.  若响应疑似噪声，则可创建短时 `provisional` 节点；只有后续再次命中或与关系边形成一致结构时才固化。
4.  对重复出现的一维或二维特征，即使复用了同一个 GmemI 节点，也必须写入新的观察实例或关系边，避免把复杂轮廓压缩成一个孤立原型。

这里要区分两个层级：

$$
\text{GmemI node}
=
\text{特征原型}
$$

$$
\text{GmemII relation / observation}
=
\text{该原型在当前结构中的位置、方向、角色和时间}
$$

### 6.4 二维连续特征的面域自指关系

一维连续特征可以用自指边压缩：同一个边缘原型沿 $\theta$ 与 $\theta+\pi$ 两个方向延伸，系统只需记录切向、步长、端点和置信度。二维连续特征也应沿用这个思想，但它的延伸方向不是两个离散方向，而是一组连续角度扇区。

一维自指边可抽象为：

$$
e_{self}^{1D}
=
\left(
n_i\rightarrow n_i,\,
support\_dim=1,\,
\hat{\mathbf{t}},\,
s,\,
endpoint_+,\,
endpoint_-,\,
confidence
\right)
$$

二维 surface 的自指关系应升级为“面域外延自指关系”：

$$
e_{self}^{2D}
=
\left(
n_i\rightarrow n_i,\,
support\_dim=2,\,
anchor,\,
\Theta_i,\,
\rho_i^{min}(\theta),\,
\rho_i^{max}(\theta),\,
C_i(\theta),\,
boundary_i(\theta),\,
holes_i,\,
confidence
\right)
$$

其中：

*   $\Theta_i$ 是该 surface 已观察到的连续角度扇区集合。
*   $\rho_i^{min}(\theta),\rho_i^{max}(\theta)$ 是从 anchor 出发，在方向 $\theta$ 上同类 surface 的内外径范围。
*   $C_i(\theta)$ 是该方向上同类 surface 连续性的置信度。
*   $boundary_i(\theta)$ 记录该方向最终遇到的是边缘、颜色过渡、相邻 surface、无效视野还是未知 frontier。
*   $holes_i$ 记录孔洞候选或内边界关系。

工程上不需要连续函数精确表示，可以把 $\theta$ 离散成 $K_\theta$ 个角度 bin，或用少量连续扇区区间表示：

$$
\Theta_i
=
\{[\theta_1,\theta_2],[\theta_3,\theta_4],\dots\}
$$

当窗口关键点 $\mathbf{k}$ 被判定为同一 surface 原型 $n_i$ 时，以 GmemII anchor $\mathbf{a}$ 为参照：

$$
\Delta\mathbf{p}
=
\mathbf{k}-\mathbf{a},
\quad
\rho=\|\Delta\mathbf{p}\|,
\quad
\theta=\operatorname{atan2}(\Delta y,\Delta x)
$$

更新对应角度扇区：

$$
\rho_i^{max}(\theta)
\leftarrow
\max
\left(
\rho_i^{max}(\theta),
\rho
\right)
$$

$$
C_i(\theta)
\leftarrow
(1-\alpha)C_i(\theta)
+
\alpha S_i^I(\mathbf{k})
$$

若相邻角度 bin 的 $C_i(\theta)$ 连续高于阈值，则合并为一个扇区；若中间出现强边界或强异质 surface，则切分扇区。这样，大面积同色区域不会生成大量重复关系边，而是被压缩为“同一 surface 在一组连续方向上延伸到某个外径”的结构。

面域自指关系的写入触发条件应是结构性变化，而不是每个同色采样点：

1.  新增未覆盖角度扇区。
2.  某个方向的 $\rho_i^{max}$ 增长超过 $\theta_{\rho}^{expand}$。
3.  颜色均值、方差或纹理统计出现稳定残差。
4.  该方向遇到边缘、过渡带、孔洞、交界或无效视野。
5.  GposII/GposI 对该 surface 的解释域发生明显扩张或分裂。

该结构仍然遵守图表征约束：GmemI 节点保存 surface 原型，GmemII 中的自指关系保存该原型在当前对象中的二维外延关系；真正的复杂边界形状仍由边缘、过渡带、角点和关系边补足，而不是把像素 mask 直接塞进节点。

### 6.5 GmemII 开启

当当前窗口中存在稳定关键点，且没有正在学习的 GmemII，立即开启一个临时 GmemII：

$$
h_t^{work}
=
\left(
anchor,\ child\_nodes,\ relation\_edges,\ status=provisional
\right)
$$

锚点选择原则：

1.  若窗口中存在稳定二维 surface，优先选择离注视点最近且置信度最高的 surface 节点为 anchor。
2.  若只有一维结构，选择当前命中的边缘/线域节点为 anchor，并把第一段切向作为工作关系边。
3.  若只有点事件，先创建短时 GmemII；后续若没有形成边或面关系，则退化为孤立 GmemI 事件节点。

开启后立即记录：

$$
e_{anchor,j}
=
\left(
anchor\rightarrow n_j,\,
\Delta\mathbf{p}_{anchor,j},\,
support\_dim_j,\,
role_j,\,
confidence_j
\right)
$$

这样 GmemII 从第一步开始就是正在生长的结构，而不是 MICRO 结束后才批量生成的容器。

### 6.6 GmemII 填充

当存在当前工作 GmemII $h$ 时，每次注视执行：

1.  将窗口关键点对应的 GmemI 节点加入 $h.child\_nodes$。
2.  写入 anchor 到关键点的关系边。
3.  写入上一注视窗口关键点到当前关键点的时间相邻边。
4.  对一维连续结构，更新工作边：
    $$
    e^{work}
    =
    (src,\ dst,\ \Delta\mathbf{p},\ \hat{\mathbf{t}},\ age,\ confidence)
    $$
5.  对二维 surface，更新覆盖图 $O_h^{surf}$、颜色均值、颜色方差、surface frontier 和 $e_{self}^{2D}$ 的角度扇区外延。
6.  若二维 surface 的新关键点没有带来扇区扩张、统计残差或边界事件，则只更新统计，不新增关系边实例。
7.  对点事件，写入 `corner`、`junction`、`terminator` 等角色关系。

关系边合并只在相对位置接近时发生：

$$
d_{\Sigma}
\left(
\Delta\mathbf{p}_{new},
\Delta\mathbf{p}_{old}
\right)
<
\theta_{edge}
$$

否则应保留为多重边，避免同一对特征原型的不同空间关系互相覆盖。

### 6.7 GmemII 闭合、挂起与恢复

GmemII 不应只依赖固定步数闭合，而应由完成度和 frontier 共同决定。

闭合条件：

$$
C_h>\theta_{complete}
\quad
\land
\quad
F_h^{remain}<\theta_{frontier}
\quad
\land
\quad
N_W<\theta_{novel}^{close}
\quad
\land
\quad
G_h<\theta_{gap}^{close}
$$

其中 $F_h^{remain}$ 是当前 GmemII 尚未访问 frontier 的总量，$G_h$ 是未闭合关系缺口。

对一维结构，还应满足：

1.  两端均被端点、交点、闭环或边界外消失证据解释。
2.  工作边连续失败超过阈值，但不是由于局部噪声导致的一步失败。

对二维 surface，还应满足：

1.  内部统计稳定：
    $$
    \Delta \mu_{rgb}<\theta_{\mu}
    \quad
    \land
    \quad
    \Delta \sigma_{rgb}<\theta_{\sigma}
    $$
2.  $e_{self}^{2D}$ 中高置信角度扇区的 $\rho^{max}(\theta)$ 不再显著增长。
3.  surface frontier 已转化为边界、过渡带、相邻 surface，或已离开有效视野。
4.  若存在 $holes_i$，内边界已被边缘、过渡带或无效区域证据解释；否则 GmemII 只能标记为未完成或含孔洞候选。

挂起条件：

1.  REVIEW 抢占发生。
2.  当前局部向量多步失败，但 $F_h^{remain}$ 或 $G_h$ 仍高。
3.  当前窗口出现更高优先级的新结构，但原 GmemII 尚未完成。

恢复条件：

$$
V_{resume}>\theta_{resume}
$$

或未来 GposII/GposI 响应重新命中该未完成 GmemII 的已观察子结构。

---

## 7. 简化算法流程

推荐最小可实现流程如下：

```text
Input:
    当前注视点 p_t
    感官特征场 X_t
    GmemI, GmemII, GmemIII
    当前状态 state_t
    当前工作 GmemII h_work
    动态矩阵 P_t

Step 1: 选择动态视窗口
    构造 R_work = 当前 Gmem/Gpos 响应域 + frontier + novelty
    生成候选面积预算 A = logspace(pi*r_min^2, pi*r_max^2, K)
    对每个 A_k 执行 seeded_grow(R_work, barrier, A_k) 得到 W_k
    根据响应域一致性、Gpos 响应尺度、细节残差、粗结构连贯度、frontier 触边和多对象混合计算 Z_k
    平滑得到不规则工作域 W_t 和等效半径 r_eff

Step 2: 实时计算 Gpos
    计算 W_t 内活跃 GmemI 的 GposI 响应
    计算或更新 GposII/GposIII 的全局或低分辨率峰值

Step 3: 动态状态竞争
    计算 N_W, F_W, Q_W, C_h, G_h, R_out^{II/III}, E_fail
    计算 z_macro, z_micro, z_review
    若 REVIEW 抢占成立:
        suspend 当前上下文
        state = REVIEW_VERIFY
        p_target = q_review*
        执行眼跳
        return

Step 4: 窗口关键点采样
    生成 known_confirm, known_extend, novel_seed, event_peak 候选
    过滤 size < ell_min_d(W_t) 的普通细节候选
    阈值筛选 + NMS
    每个支持维度最多取 K_d 个关键点

Step 5: Gmem 写入
    若无 h_work 且 MICRO 分数高:
        创建 provisional GmemII
    对每个关键点:
        匹配或创建 GmemI
        写入 observation relation
        更新工作关系边、coverage、frontier
        若为二维连续 surface，更新 e_self^{2D} 的连续角度扇区与径向外延

Step 6: 更新动态矩阵
    P_{t+1} = decay(P_t) + edge butterfly + surface shell/frontier + event + relation gap - return inhibition

Step 7: 合成局部眼动向量
    根据 W_t 的有效面积和 r_eff 设置 s_min, s_max, edge step 和 surface dilation
    v_t = lambda_P v_P + lambda_low v_lowGpos + lambda_gap v_gap
          + lambda_nov v_novel + lambda_inertia v_{t-1} - lambda_return v_return

Step 8: 执行或退出
    若 ||v_t|| 足够:
        p_{t+1} = p_t + ClipStep(v_t)
    否则:
        E_fail += 1
        由下一步状态竞争决定 MACRO、REVIEW 或挂起

Step 9: 闭合判断
    若 h_work 满足完成条件:
        h_work.status = completed
        state = MACRO_SEARCH
```

---

## 8. 与现有优化器的优劣对比

| 维度 | 全局 drive 路线 | 视窗口向量路线 |
| --- | --- | --- |
| MICRO 连续学习 | 容易受全图最大值干扰 | 更适合沿局部流形延展 |
| MACRO 搜索 | 强 | 需要全局机制辅助 |
| REVIEW 结构召回 | 强，尤其依赖 GposII/GposIII | 通过抢占机制接入 |
| 大块 surface | 可能在内部打转 | 可由 frontier 推向外沿 |
| 一维边缘追踪 | 需要额外意图或强切向门控 | 可由蝴蝶核和工作边自然形成 |
| 参数复杂度 | 权重多，竞争尺度不一致 | 局部参数较少，但需要状态竞争 |
| 计算复杂度 | 全图矩阵叠加较重 | MICRO 较轻，REVIEW 仍需结构峰值 |
| 风险 | argmax 误导、面积优势、噪声峰 | 局部困住、断裂处恢复困难 |

最终建议是采用混合结构：

1.  MICRO 以视窗口向量为主。
2.  MACRO 保留全局或低分辨率探索。
3.  REVIEW 保留 GposII/GposIII 的结构响应抢占。
4.  GposI 每步服务于局部写入和低非零采样。
5.  GposII/GposIII 每步服务于状态竞争和外部结构召回，而不是直接支配 MICRO 的每一步。

---

## 9. 风险与控制

### 9.1 局部困住

若当前窗口内所有方向近似对称，$\mathbf{v}_t$ 会趋近零。处理方式不是加入随机噪声，而是提高 $E_{fail}$，让状态竞争切到 MACRO。

### 9.2 边缘虚假延展

蝴蝶核可能沿已观察边缘方向预测出不存在的延续。控制方式是让真正执行向量时乘以低阈值 GposI 或原始边缘响应，要求：

$$
S_{edge}^{I}(\mathbf{q})>\theta_{edge}^{low}
\quad
\text{or}
\quad
R_{edge}(\mathbf{q})>\theta_{raw}^{low}
$$

若预测方向连续失败，则写入端点或终止关系，而不是继续外推。

### 9.3 surface 泄漏

二维 frontier 可能越过弱边界进入相邻 surface。控制方式是用边缘、过渡带或颜色差作为导通阻断：

$$
F_{h,i}^{surf}
\leftarrow
F_{h,i}^{surf}
\odot
\left(
1-B_{barrier}
\right)
$$

其中 $B_{barrier}$ 来自 grad、RGB_TRANSITION 或颜色残差。

### 9.4 状态震荡

动态竞争若没有迟滞，会在 MICRO/REVIEW/MACRO 间抖动。必须保留：

1.  最小驻留时间 $T_{min}$。
2.  切换边际 $\Delta_{switch}$。
3.  REVIEW 最近失败抑制 $E_{review\_recent}$。
4.  挂起栈的恢复价值阈值 $\theta_{resume}$。

### 9.5 尺度震荡与细节遗忘

动态视窗口可能在粗尺度和细尺度之间震荡，也可能在大窗口下长期忽略细节。控制方式是：

1.  对 $\log r_t^{eff}$ 做平滑和最大变化限制。
2.  将小尺度残差写入 `detail_pressure`，若同一区域多次产生稳定细节压力，则强制生成一次细节观察。
3.  维护尺度分层访问抑制，避免粗尺度覆盖错误压制细尺度探索。
4.  对大窗口写入设置最低尺度阈值，避免把细节噪声错误固化为整体结构的一部分。

---

## 10. 暂定结论

视窗口向量路线比纯全局 drive 更适合作为 MICRO 的主机制。它把“一维沿端点延展、二维向外沿扩展、点事件吸引、关系缺口补全”统一为局部向量合成，而不是机械拆分为多个 MICRO 意图。

但该路线不应完全替代全局机制。更合理的整体架构是：

$$
\text{MACRO: global selection}
\quad
\text{MICRO: local vector}
\quad
\text{REVIEW: structural preemption}
$$

其中 GposI/GposII/GposIII 每步实时响应，分别承担局部特征定位、结构召回和状态抢占。这样既能保留图结构的再现与联想能力，也能降低 MICRO 在复杂边缘和大块连续二维特征上反复打转的风险。
