# 工程文档：基于全局兴趣图的扫视策略与局部连续性学习

## 1. 目标
本设计用于将视觉扫视过程组织为一个可增量学习、可终止判定、可回溯重置的工程系统。系统核心由三部分组成：

1. **全局兴趣图（Interest Map）**：为当前图像维护一张动态的兴趣分布图，用于决定下一注视点。
2. **局部连续性模块（Texture / Edge Continuity）**：仅负责判定当前位置周围的纹理连续性和边缘连续性，不参与全局兴趣计算。
3. **Episode 调度器**：根据全局兴趣图、局部连续性、访问轨迹和完成度决定下一注视点或结束当前 episode。

该方案的目标是：
- 提高扫视路径的局部一致性与对象内覆盖率；
- 降低跨对象错误学习；
- 让 graph1 与 graph2 的学习更符合分层、分时的增量机制；
- 保持 GPU 友好，尽量通过预计算和张量操作完成。

---

## 2. 总体架构
系统按“静态缓存 + 动态 episode 状态 + 局部查询”三层组织。

### 2.1 静态缓存（每张图只计算一次）
在图像输入后立即预计算以下特征图：
- `grad_map`：梯度幅值图
- `hue_map`：色相图
- `cur_map`：曲率图
- `asp_map`：长宽比图
- `ori_map`：方向图
- `texture_base_map`：纹理连续性基础图
- `edge_base_map`：边缘连续性基础图
- `texture_cc_map`：纹理连通域标签图
- `edge_fragment_graph`：边缘片段图

### 2.2 动态 episode 状态
每个 episode 维护：
- `interest_map`：全局兴趣图
- `completed_map`：已访问完成区域图
- `trace_map`：轨迹抑制图（用于“刚扫过兴趣最低，随后缓慢恢复”）
- `visited_mask`：是否访问过
- `last_visit_time_map`：每个位置最近访问时间
- `current_fixation`：当前注视点
- `episode_step`、`episode_time`

### 2.3 局部查询
每次 fixation 只执行：
- `_compute_texture_resp(point)`：计算纹理连续性
- `_query_edges(point, radius)`：查询局部边缘连续性
- `next_point(...)`：综合兴趣图与连续性决定下一点

---

## 3. 全局兴趣图设计

### 3.1 设计原则
全局兴趣图需要满足以下性质：
1. 当前注视点的兴趣为 0。
2. 注视点周围兴趣随半径先上升后下降。
3. 刚扫过的路径兴趣最低，随后随时间缓慢恢复。
4. 已访问完成的区域兴趣持续压低。
5. 支持 reset。

### 3.2 基础兴趣项
基础兴趣图由纹理连续性和边缘连续性构成：

\[
B(x)=w_t T(x)+w_e E(x)
\]

其中：
- `T(x)`：纹理连续性基础分数；
- `E(x)`：边缘连续性基础分数；
- `w_t, w_e`：权重。

### 3.3 环形注视抑制项
为实现“当前点为 0，附近先升后降”，定义环形核：

\[
R(d)=\frac{d}{\sigma_r}\exp\left(-\frac{d^2}{2\sigma_r^2}\right)
\]

其中 `d` 是到当前注视点的距离。该核在 `d=0` 时为 0，在中等距离处达到峰值。

### 3.4 轨迹抑制与恢复项
使用最近访问时间图 `last_visit_time_map` 构造抑制项：

\[
H(x,t)=\exp\left(-\frac{t-t_{last}(x)}{\tau_{rec}}\right)
\]

并定义：

\[
M_{trace}(x,t)=1-\lambda_{trace} H(x,t)
\]

这样刚访问过的位置被压低，之后随时间逐渐恢复。

### 3.5 已完成区域抑制项
已完成区域图 `completed_map` 直接用于压低兴趣：

\[
M_{comp}(x)=1-\lambda_{comp}\cdot completed(x)
\]

若某位置已完成，则其兴趣接近 0。

### 3.6 最终兴趣图
综合为：

\[
I_t(x)=\text{Norm}\Big(B(x)\cdot R(d(x,f_t))\cdot M_{trace}(x,t)\cdot M_{comp}(x)\Big)
\]

其中 `f_t` 为当前注视点。

### 3.7 Reset 机制
- `reset_image(image_id)`：新图像输入时，清空动态状态并重建静态缓存。
- `reset_episode()`：同图像继续扫描但重新开始一个 episode，只清空动态状态，保留静态缓存。

---

## 4. `_compute_texture_resp()` 设计

### 4.1 职责
仅负责计算当前位置周围的**纹理连续性**，不参与兴趣图计算，也不参与 episode 终止判断。

### 4.2 输入
- `point`
- 静态缓存：`grad_map`, `hue_map`, `cur_map`, `asp_map`, `ori_map`

### 4.3 计算思路
先在局部邻域提取多模态 patch，再分别计算子模态的一致性：
- `s_grad`：梯度连续性
- `s_hue`：色相连续性
- `s_cur`：曲率连续性
- `s_asp`：长宽比连续性
- `s_ori`：方向连续性

可定义各模态的局部一致性为：

\[
 s_m(p)=\exp\left(-\frac{Var(m|N_r(p))}{\sigma_m^2}\right)
\]

方向模态可用角度一致性单独计算。

### 4.4 模态融合
纹理连续性定义为：

\[
T(p)=\prod_m s_m(p)^{w_m}
\]

该形式强调：任一模态严重不一致时，纹理连续性会下降。

### 4.5 输出
输出单标量 `texture_resp ∈ [0,1]`。

### 4.6 GPU 要点
- 所有模态图先在 GPU 上完成全图预计算；
- 局部 patch 通过 `unfold` / `gather` 抽取；
- 使用张量运算计算方差、均值和方向一致性；
- 不要在 Python 层按点循环计算。

---

## 5. `_query_edges()` 设计

### 5.1 职责
仅负责局部边缘连续性查询，不参与全局兴趣计算。

### 5.2 推荐输出
建议返回边缘片段候选，而不是裸边缘像素。每个候选片段至少包含：
- `fragment_id`
- `centroid`
- `tangent_direction`
- `length`
- `visited_ratio`
- `closure_status`（开链/半闭/成环）
- `neighbors`

### 5.3 边缘连续性定义
对边缘片段的局部连续性可以由梯度强度、方向一致性、曲率平滑性、局部形状一致性综合得到：

\[
E(p)=w_g s_{grad}(p)+w_o s_{ori}(p)+w_c s_{cur}(p)+w_a s_{asp}(p)
\]

### 5.4 实现方式
边缘应在静态预计算阶段形成：
- `edge_prob_map`
- `edge_orientation_map`
- `edge_fragment_graph`

`_query_edges(point, radius)` 只负责从局部窗口中提取相交的 fragment，并返回候选列表。

### 5.5 GPU 要点
- 边缘片段图预构建一次；
- query 时只做索引、mask、gather 与 top-k；
- 不要在每次 fixation 中重新做连通域分析。

---

## 6. 下一注视点选择

### 6.1 候选生成
从兴趣图中找局部峰值，步骤如下：
1. 对 `interest_map` 做局部非极大值抑制；
2. 排除当前注视点附近区域；
3. 排除 `completed_map` 中已完成区域；
4. 排除最近访问过的点。

### 6.2 候选评分
每个候选点的综合得分：

\[
S(c)=\alpha I_t(c)+\beta Q_{cont}(c)-\gamma D(c,f_t)-\delta P_{revisit}(c)
\]

其中：
- `I_t(c)`：全局兴趣图值；
- `Q_{cont}(c)`：连续性质量；
- `D(c,f_t)`：眼跳距离成本；
- `P_{revisit}(c)`：重复访问惩罚。

### 6.3 模式切换
- **纹理模式**：当局部边缘较弱时，优先沿纹理连通域移动；
- **边缘模式**：当局部边缘存在时，优先沿边缘片段跟踪。

### 6.4 停止条件
可用以下条件之一终止当前 episode：
1. 候选最高分低于阈值；
2. 当前局部已完成；
3. 边缘或纹理已经走完；
4. 新信息收益低于阈值；
5. 达到最大步数或最大时间。

---

## 7. Episode 运行流程

### 7.1 初始化
- `reset_episode()`
- 设置初始 `current_fixation`
- 生成初始 `interest_map`

### 7.2 每次注视
- 调用纹理连续性计算；
- 调用边缘连续性查询；
- 更新 `graph1`；
- 更新 `trace_map` 与 `visited_mask`；
- 必要时更新 `completed_map`。

### 7.3 下一点选择
- 从 `interest_map` 找候选；
- 与连续性、眼跳距离一起打分；
- 选择得分最高者作为下一点。

### 7.4 Episode 结束
- 若终止条件满足，则结束当前 episode；
- 统一进入高层学习（graph2 批处理更新）。

---

## 8. 代码模块划分建议

### 8.1 AttentionPolicy
负责调度：
- `reset_image()`
- `reset_episode()`
- `update_interest_map()`
- `next_point()`
- `should_end_episode()`

### 8.2 TextureAnalyzer
负责纹理静态缓存与局部连续性：
- `precompute(image)`
- `_compute_texture_resp(point)`

### 8.3 EdgeAnalyzer
负责边缘静态缓存与局部连续性：
- `precompute(image)`
- `_query_edges(point, radius)`
- `_next_edge_point(current_point, ...)`

### 8.4 EpisodeMemory
负责 episode 内动态状态：
- `update_trace(point, time)`
- `update_completed(region)`
- `query_completed(point)`

---

## 9. GPU 加速原则

1. **密集计算前置**：所有模态的基础图先统一预计算。  
2. **局部查询张量化**：局部连续性只做 `gather` / `unfold` / `topk`。  
3. **结构图预构建**：边缘连通结构不要每次重建。  
4. **避免 Python 循环**：尤其不要在每个 fixation 内做像素级扫描。  
5. **访问痕迹用张量**：`trace_map` 和 `completed_map` 应以张量形式存储并更新。

---

## 10. 默认参数建议

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `sigma_r` | 0.08 * 图像对角线 | 环形兴趣核半径尺度 |
| `tau_rec` | 10~30 steps | 轨迹恢复时间常数 |
| `lambda_trace` | 0.7 | 轨迹抑制强度 |
| `lambda_comp` | 1.0 | 完成区域抑制强度 |
| `C_thresh` | 0.85 | 完成区域覆盖阈值 |
| `G_thresh` | 0.01 | 新信息增益阈值 |
| `max_steps` | 20 | 单 episode 最大步数 |
| `edge_search_radius` | 40 px | 边缘局部查询半径 |

---

## 11. 验证标准

### 11.1 功能验证
- 当前点的兴趣为 0；
- 当前点附近兴趣呈先升后降的环形分布；
- 刚扫过的路径兴趣低，随后逐渐恢复；
- `reset_image()` 后静态缓存与动态状态均重置；
- `_compute_texture_resp()` 和 `_query_edges()` 不直接参与全局兴趣计算。

### 11.2 行为验证
- 在无明显边缘区域，注视点应沿纹理连通域移动；
- 在边缘显著区域，注视点应沿边缘片段连续前进；
- 当局部区域已完成时，应停止在该区域继续回扫。

### 11.3 性能验证
- 静态缓存重算只发生在新图像输入时；
- 每次 fixation 的开销应主要由局部张量运算组成；
- 连接域和边缘图不应在每次 fixation 中重复构建。

---

## 12. 实施顺序建议

1. 先实现 `trace_map`、`completed_map` 与 `interest_map`。  
2. 再实现 `_compute_texture_resp()` 与 `_query_edges()` 的静态缓存化。  
3. 然后实现 `next_point()` 的综合打分。  
4. 最后接入 episode 终止判定与 graph2 批量学习入口。

---

## 13. 结论
本方案将扫视过程拆分为“静态预计算、局部连续性查询、动态兴趣驱动选择、episode 结束后统一学习”四个环节。其优点是：
- 利于 GPU 加速；
- 利于多层级学习；
- 便于加入对象内连续扫描与边界跟踪；
- 能够为 graph1 / graph2 的分层学习提供稳定调度机制。

如果后续需要扩展到多层级金字塔结构，可在本方案基础上增加 `MultilevelCoordinator`，把每个层级都视为一个独立的 episode 子过程，并以兴趣图与完成图作为统一入口/出口条件。

