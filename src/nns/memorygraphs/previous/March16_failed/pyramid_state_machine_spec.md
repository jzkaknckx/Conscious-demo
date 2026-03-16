# 技术文档：基于分层/扫视驱动的增量学习 — 所需代码修改与新增模块清单

目的：把扫视/注视驱动学习逻辑（局部只做 graph1 快速学习；在一次扫视结束后批量更新 graph2；基于纹理连通与边缘跟踪选择下一个注视点；多层级金字塔式检索/学习状态机）落地到现有代码库。下面以 **“按类/方法逐条”** 的方式列出需要修改或新增的代码块（包括接口、行为、伪代码、默认参数、验证要点与测试建议）。文档按优先级排列，先核心（Saccade 类），再关联模块（GraphI / GraphII / 支撑模块），最后集成/测试/监控。

> 说明：每一处修改都给出**目的 / 输入输出 / 核心伪代码 / 默认参数 / 复杂度/注意点 / 测试要点**，便于直接在代码中实现。

---

# 目录（快速导航）

1. Saccade 类 — 重构与新增方法（核心）
2. EpisodeManager / STM（短时记忆缓冲） — 新增模块
3. TextureChannel 与 EdgeContourTracker — 新增或替换的感知通道
4. HypothesisManager / Slot-binding — 新增（对象假设绑定与完成判定）
5. GraphI（graph1） — 修改：只在 fixation 时更新（fast plasticity）
6. GraphII（graph2） — 修改：支持批量/统一更新接口（slow consolidation）
7. SelectorPolicy（注视点选择策略） — 新增（纹理/边缘优先策略）
8. MultilevelCoordinator（多层级金字塔状态机） — 新增（多层检索/学习入口）
9. 接受标准（Acceptance criteria）

---

# 1. Saccade 类 — 必要修改与新增方法（核心入口）

**目的**：把扫视逻辑从“串行 graph1→graph2 每帧执行”改为“每 fixation 只做 graph1；一次扫视结束由 Saccade 调用 EpisodeManager 批量触发 graph2 更新；支持基于纹理/边缘的下一个注视点选择；支持多层级入口/出口接口。**

**要修改或新增的字段（属性）**

* `state`：枚举(`wander`, `learn`, `end`, `exploit`, `idle`)（保留旧状态并支持扩展）
* `current_fixation`：{x,y, frame_id, time, features_snapshot}
* `fixation_history`：队列（固定长度，最近 N fixations）
* `episode_id`：当前 episode 标识
* `episode_buffer`：引用 EpisodeManager（或直接持有短期 buffer）
* `saccade_matrix`：方形稀疏矩阵 / 布尔网格，用作辅助记忆（记录在 episode 中每个格子是否被观察）
* `layer_level`：当前工作层级（整数，用于多层级协调）
* 配置参数（示例）：`max_fixations_per_episode=20`, `dwell_time_max=1.5s`, `texture_window_radius=R_t=32px (norm)`, `edge_search_radius=R_e=40px (norm)`

**新增 / 修改方法（含伪代码与说明）**

1. `start_episode(self, initial_point, layer_level=top)`
   目的：初始化 episode 上下文，在 episode_buffer 中建立记录。
   输入：初始注视点、层级。
   输出：episode_id。
   伪码：

   ```python
   def start_episode(self, initial_point, layer_level=0):
       self.episode_id = EpisodeManager.create_episode(layer_level)
       self.current_fixation = initial_point
       self.fixation_history.clear()
       self.saccade_matrix.clear()
       self.state = 'learn'
   ```

2. `perform_fixation(self, image, time_now)`
   目的：对当前注视点做感知（提取 proto、texture、edge），只执行 graph1 层更新并把结果写入 STM/episode buffer。
   输入：image (或 frame), timestamp
   输出：update to EpisodeManager; returns features snapshot
   伪码（关键点）：

   ```python
   def perform_fixation(self, image, time_now):
       features = Perception.extract_features_at(self.current_fixation, image)
       texture_resp = TextureChannel.compute(self.current_fixation, image)
       edge_resp = EdgeContourTracker.sample_local(self.current_fixation, image)
       # only graph1 fast update
       GraphI.fast_update(features, location=self.current_fixation)
       # record into episode buffer
       EpisodeManager.append_observation(self.episode_id, features, texture_resp, edge_resp, time_now)
       self.fixation_history.push((self.current_fixation, time_now))
   ```

   注意：**绝不在此方法触发 graph2 的重构 / 合并**。

3. `decide_next_fixation(self, image)`
   目的：根据当前 fixation、texture/edge responses 和 episode 状态计算下一个注视点；实现你提出的规则：如果当前附近没有边缘，则沿着纹理连通域滑动；若在边缘附近则沿边缘跟踪直到结束或成环。
   输入：当前 image / per-pixel precomputed maps (edge map, texture map)
   输出：next fixation (x,y) or signal to end episode
   伪码（核心逻辑）：

   ```python
   def decide_next_fixation(self, image):
       # get local edge strength within R_e
       local_edges = EdgeContourTracker.query(self.current_fixation, radius=R_e)
       if local_edges.is_empty():
           # follow texture connected region
           next_pt = TextureChannel.next_point_along_texture(self.current_fixation, self.saccade_matrix)
       else:
           # follow connected edge fragments
           next_pt = EdgeContourTracker.next_point_along_edge(self.current_fixation, self.saccade_matrix)
       if next_pt is None or self.fixations_exceeded():
           return 'end_episode'
       return next_pt
   ```

   参数与行为说明见后文 `TextureChannel` 与 `EdgeContourTracker` 部分。

4. `end_episode(self)`
   目的：调用 EpisodeManager 批量处理（graph2 的统一更新），清理 episode。
   伪码：

   ```python
   def end_episode(self):
       completed_hypotheses = EpisodeManager.finalize_episode(self.episode_id)
       # batch update graph2 for completed hypotheses only
       for hyp in completed_hypotheses:
           GraphII.batch_update_from_hypothesis(hyp)
       self.state = 'idle'
   ```

   注意：EpisodeManager.finalize_episode 应负责“完成判定”（coverage, marginal_gain）并返回仅完成的 hypotheses。

5. `global_search_for_initial_fixation(self, image)`
   目的：（权宜之计）当需要新起点且 graph2 中没检索到匹配时，扫描图像并选择 edge gradient 最大点或候选热点（edge-based）。
   输出：point coordinates

**测试要点（Saccade）**

* 单次 fixation 仅触发 GraphI.fast_update，GraphII 不改变（验证通过）。
* 在边缘稀少区域，连续 decide_next_fixation 应沿纹理区域移动（可在合成图测试）。
* 在边缘存在情况下沿边缘连通方向行走直到边缘结束或成环（记录 saccade_matrix 覆盖）。
* end_episode 必触发 EpisodeManager.finalize 并且 GraphII.batch_update_from_hypothesis 被调用一次。

---

# 2. EpisodeManager / STM（短时记忆缓冲） — 新增模块

**目的**：为一个扫视 episode 维护短期记忆（STM），收集 fixation 期间的 proto / texture / edge observations、建立 object hypotheses、做完成判定、并在 finalize 时输出要合并到 graph2 的 hypotheses。

**主要接口（建议实现）**

* `create_episode(layer_level) -> episode_id`
* `append_observation(episode_id, fixation_point, features, texture_resp, edge_resp, timestamp)`
* `update_hypotheses(episode_id)` — 每次 append 后调用，用 data association 规则把 obs 绑定到 hypotheses
* `finalize_episode(episode_id) -> list[completed_hypothesis]` — 执行 completion 判定并返回完成的 hypotheses (each hypothesis is summary struct)
* `get_episode_summary(episode_id)` — 返回封装数据（用于 logging / debugging）
* `abort_episode(episode_id)` — 清理

**数据结构（per episode）**

* `episode_id, layer_level, start_time, end_time, fixation_list[]`
* `hypotheses[]` where each hypothesis has:

  * `hyp_id, proto_ids_set, centroid_stats, coverage_estimate, times_seen, confidence, member_fixations[]`
  * `estimated_total_proto` (used by coverage metric)
  * `state` (`incomplete`, `completed`, `promoted`)
* `saccade_matrix` (grid aligned with image or relative grid) used to check local coverage & to avoid re-fixating same subregion

**核心算法（association & completion）**

* Association: when append obs, compute association score to existing hypotheses = weighted(sum(spatial proximity, feature similarity, temporal contiguity)). If score ≥ assoc_thresh assign; else create new hypothesis. Default thresholds: `assoc_thresh = 0.55`. Spatial proximity uses normalized distance; feature similarity uses bitset overlap/descriptor cosine.
* Coverage estimation: `coverage = observed_proto_count / estimated_total_proto` where `estimated_total_proto` inferred via growth curve (e.g., Chao1 estimator) or a heuristic `max(observed, 1.2 * observed)`.
* Completion condition: coverage ≥ `C_thresh` (default 0.85) OR marginal info gain ≤ `G_thresh`.

**Outputs**

* For each completed hypothesis produce a compact summary to give GraphII: `{proto_histogram, proto_centroids (top-M), pair_tokens, first_seen, last_seen, confidence}`.

**Testing**

* Simulated episode with known object: ensure finalize returns one completed hypothesis when coverage reached.
* Edge cases: partial coverage -> hypothesis remains incomplete & not sent to GraphII.

---

# 3. TextureChannel 与 EdgeContourTracker — 感知通道（新增或替换）

这两部分实现你关于“纹理卷积连通域”和“边缘连通跟踪”的注视策略。

## TextureChannel（新增）

**职责**

* 对输入图像建立一个 texture response map（纹理强度/方向/局部连通域指示）用于在无明显边缘时沿连通域移动 fixation。
* 提供方法：`compute_texture_map(image)`（可离线/全图），`next_point_along_texture(current_point, saccade_matrix)`（在线调用）。

**实现思路**

* 纹理卷积器：使用一组 Gabor / steerable filters 或自定义小卷积核集以捕捉局部纹理方向与连通性。
* 连通域提取：对 texture response threshold 后做连通组件检测（connected components）。维护组件标签图 `tex_cc_label[y,x]`.
* `next_point_along_texture`：在当前 component 内随机或基于 geodesic distance 选择尚未被 saccade_matrix 覆盖的点（优先边界处或最大不确定位置）。

**接口示例**

* `TextureChannel.compute_texture_map(image) -> texture_map, cc_label_map`
* `TextureChannel.next_point_along_texture(current_point, saccade_matrix) -> next_point | None`

**默认参数**

* filter bank scales: [3,6,12 px], orientation bins 8
* texture threshold: 0.15 (normalized)
* preferred next point selection: maximize local entropy or distance from previous fixations

## EdgeContourTracker（新增/扩展）

**职责**

* 提取边缘强度/连通边缘片段（contour fragments）；在当前注视附近找到并追踪最近的 edge fragment，支持 `next_point_along_edge`。
* 提供 `query(current_point, radius)` 返回 local edge fragments; `next_point_along_edge(current_point, saccade_matrix)` 返回沿 fragment 下一个点或 None.

**实现思路**

* Use Canny / structured edge for global map, then do contour extraction -> obtain polylines.
* For each polyline maintain adjacency & visited flags (per episode via saccade_matrix).
* next_point selects next vertex along polyline in consistent direction until polyline ends or loops.

**接口**

* `EdgeContourTracker.extract_edges(image) -> edge_map, polylines`
* `EdgeContourTracker.query(point, radius) -> list(polylines_overlapping)`
* `EdgeContourTracker.next_point_along_edge(point, saccade_matrix) -> next_point | None`

**参数**

* edge thresholding: adjustable; default use Canny thresholds tuned to domain.

**Testing**

* On synthetic images: ensure next_point follows the contour; ensure loop detection works; ensure polyline end detection.

---

# 4. HypothesisManager / Slot-binding — 新增模块（可选但强烈推荐）

**目的**：提供“slot”机制（有限数目的 object slots）用于在一个 episode 内绑定观察到的局部特征到对象假设，便于稳定的 association。推荐采用简单 slot data structure 而不是全训练 Slot Attention 初版（后者可替代）。

**功能**

* `create_slot()`, `assign_obs_to_slot(slot_id, obs)`, `merge_slots(slot_id_a, slot_id_b)`, `slot_completion_check(slot_id)`, `export_completed_slots()`.

**实现细节**

* Slot 内记录 proto histogram、centroid distribution、history of matches。合并/分裂策略按 temporal cooccurrence.

**参数**

* num_slots default = 32 (tunable).

**测试**

* Ensure slot assignment stable across sequential fixations on same object; check merge triggers when two slots show high cooccurrence.

---

# 5. GraphI（graph1） — 修改：仅在 fixation 执行 fast updates

**变更点**

* 将现有 graph1 的更新路径改为可被 `GraphI.fast_update(features, location)` 调用的轻量 API。
* graph1 内部增加 `fast_update`（即时、局部）与 `slow_summarize`（用于 episode-level aggregation, optional）两个方法。

**fast_update 要求**

* 快速插入/增强 local proto counts、update centroid accumulators、update last_seen
* O(1) 或 O(k) 时间（k = num protos in patch）
* 不执行 expensive merges / consolidation

**new API**

* `GraphI.fast_update(features, location)`
* `GraphI.get_local_summary(episode_region) -> proto_histogram, top_protos, centroids` (used by EpisodeManager when finalizing)

**Testing**

* Single fixation triggers only `fast_update` effects (counts change), no graph2 writes.

---

# 6. GraphII（graph2） — 修改：支持 batch update 接口 & soft-merge

**新增/修改接口**

* `GraphII.batch_update_from_hypothesis(hyp_summary)` — 接收 EpisodeManager 输出的 hypothesis summary，执行 slow consolidation.
* `GraphII.soft_merge(candidate_summary)` — 如果 candidate matches existing node above high threshold, enhance; if matches in mid-range, create new C-node with references (soft merge); if low, create new node.

**行为规范**

* Batch update must be idempotent (replay safe).
* Use slow learning rate; update stats using exponential moving average to avoid abrupt structural change.
* Record provenance: for every modification store `merge_history` entry listing episode_id, timestamp, inputs.

**Soft-merge logic (pseudocode)**

```python
def batch_update(candidate):
    matches = search_candidates(candidate)  # use bitset/jaccard/learned bits
    if matches.top_score >= promote_high:
        enhance_node(matches.top_node, candidate, lr=graph2_lr)
        return matches.top_node.id
    elif matches.top_score >= promote_low:
        new_node = create_node_from(candidate)
        new_node.member_refs = [matches.top_node.id, ...]  # soft link
        return new_node.id
    else:
        return create_node_from(candidate)
```

**Testing**

* Replay same hypothesis twice: graph2 should be stable (idempotence / controlled growth).
* Soft-merge should keep provenance and support later split.

---

# 7. SelectorPolicy（注视点选择策略） — 新增模块

**功能**：决定下一注视点（调用 Saccade.decide_next_fixation 内的策略）。实现两个策略分支：

* `edge_following_policy`: if local edge exists => follow along edge polylines
* `texture_following_policy`: if no local edge => follow texture CC region until border

**APIs**

* `SelectorPolicy.next_point(current_point, saccade_state, image)`
* `SelectorPolicy.global_initial_point(image)` — when no candidate from graph2, choose global starting seed (edge maximum gradient or highest novelty region)

**Parameters & heuristics**

* prefer unvisited regions (via saccade_matrix)
* maximize local information gain (estimate via texture/edge entropy)

**Testing**

* On images with large texture areas but no edges, policy should produce sequence of points inside same texture CC.

---

# 8. MultilevelCoordinator（多层级金字塔状态机） — 新增模块

**目的**：实现用户提议的多层级检索/学习流程（从 highest layer down to lowest层依次检索/学习/返回），管理每层入口/出口的状态与传参。

**职责**

* Maintain `levels[]` (e.g., Level 2 = object, Level 1 = part, Level 0 = patch)
* For each level expose `enter_level(level_id, initial_point)` and `run_level_episode(level_id)`
* On success at a lower level return to higher level and strengthen node, else if failure then continue learning at that level.

**Simplified workflow**

* `MultilevelCoordinator.handle_new_view(image)`:

  1. For level in levels_high_to_low:

     * if GraphII.search_at_level(level, query_signature) found -> strengthen and continue
     * else -> call Saccade.start_episode(layer_level=level) and run episode (with SelectorPolicy) until completion; then batch update GraphII at that level and return up.

**Testing**

* Hierarchical run on synthetic multi-scale object should go down levels only as needed; ensure completed lower-level episodes result in updates and return to upper level.

---

# 9. 接受标准（Acceptance Criteria）

1. **功能正确性**

   * 每次 fixation 调用后，只发生 GraphI.fast_update（GraphII unchanged). 单元测试通过。
   * 触发 end_episode 后，EpisodeManager.finalize 返回完成 hypotheses 并 GraphII.batch_update 被调用一次。

2. **行为规范**

   * 在没有局部边缘时，连续注视点沿 texture CC（至少 3 steps）而非跳出区域。
   * 在边缘存在时，注视点沿边缘连通追踪直至 polyline end or loop detection。

3. **稳定性**

   * 对同一对象做 N 次不同 episodes 后，GraphII 结构不出现爆炸性重复节点（growth bounded & merge rate stable).

4. **回归**

   * 现有主要功能（原有检索）在新实现下结果不降低 >5%（或其它可接受阈值）在基线任务上。

---

# 附：默认超参数汇总（一表）

* `max_fixations_per_episode = 20`
* `d_assoc = 0.08` (normalized)
* `C_thresh = 0.85`, `G_thresh = 0.01`
* `assoc_thresh = 0.55`
* `promote_high = 0.7`, `promote_low = 0.4`
* `texture_filter_scales = [3,6,12]` px (or normalized equivalents)
* `edge_search_radius = 40 px` or normalized 0.1

---
