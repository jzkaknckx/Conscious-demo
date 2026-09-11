# 视窗口向量优化器技术说明

本文说明 [graph_memorypool_newoptimizer.py](graph_memorypool_newoptimizer.py) 的主要运行逻辑，依据 2026-09-09 的代码整理。算法背景见 [Alg_optimizer_beta.md](../../../docs/algorithm/Alg_optimizer_beta.md)。下文链接中的行号用于定位当前实现；代码变更后可按方法名搜索。

## 1. 类的职责与入口

`Controller` 保存行为状态、工作语义、动态窗口和探索历史，负责状态竞争、关键点写入及眼动向量计算。`InterestOptimizer` 保存基础兴趣场与期望场，提供兴趣图合成和有效视域抑制。新算法的局部向量优化主体实际位于 `Controller` 中。

外部入口是 `MultilevelCoordinator.handle_new_view()`（[L4080](graph_memorypool_newoptimizer.py#L4080)）：将输入特征整理为 `X_subspaces`，初始化基础兴趣场，调用控制器并记录视线路径。特征张量采用 `[B, C, H, W]`，注视点采用 `(x, y)`；当前控制逻辑按单个注视点运行。

`Controller.run_step()`（[L3975](graph_memorypool_newoptimizer.py#L3975)）每步依次执行：

1. 初始化运行时矩阵、限制注视点位置，并衰减近期 REVIEW 失败量。
2. 提取注视点特征，匹配或创建 GmemI 节点，投影 GposI 响应。
3. 累积语义激活输入，选择动态窗口，计算观测量和四种行为得分。
4. 选择并提交行为状态，调用对应状态处理方法。
5. 更新三层记忆节点的神经元状态，清空本步激活输入累积量。

## 2. InterestOptimizer 主要方法

| 方法与位置 | 作用及主要输出 |
| --- | --- |
| `initialize_base_interest()`，[L1243](graph_memorypool_newoptimizer.py#L1243) | 按通道属性计算强度场 `I_str`、面连续场 `I_con1`、线连续场 `I_con2`；相加后抑制无效视域并归一化为 `base_interest`。保存原图尺寸与边缘留白参数。 |
| `valid_view_bounds()`，[L1217](graph_memorypool_newoptimizer.py#L1217) | 根据原图尺寸及 `interest_margin_radius` 计算居中的有效矩形，返回 `(y1, y2, x1, x2)`，右端点不包含。 |
| `valid_view_mask()` / `suppress_invalid_view()`，[L1229](graph_memorypool_newoptimizer.py#L1229) | 生成有效视域掩膜，或把无效位置填为指定值。候选目标场通常填负值，响应场通常填零，用于抑制填充黑边。 |
| `calculate_I_map()`，[L1319](graph_memorypool_newoptimizer.py#L1319) | 合成基础兴趣、节点激活空间场、期望场和引导场，保留各分量供调试。属于保留接口，当前新 `run_step()` 的执行链不调用它。 |
| `generate_I_guide()`，[L1298](graph_memorypool_newoptimizer.py#L1298) | 从响应图构造外围差分及远离当前注视点的引导分量，供 `calculate_I_map()` 使用。 |
| `add_expectation()`，[L1391](graph_memorypool_newoptimizer.py#L1391) | 在预测位置叠加外围正、中心负的高斯差分期望场，写入 `I_exp[node_id]`。新 REVIEW 成功时仍会调用。 |
| `clear_node_fields()`，[L1415](graph_memorypool_newoptimizer.py#L1415) | 删除指定特征节点的期望场；`is_semantic=True` 时不执行删除。 |

初始化入口目前只在 `base_interest is None` 时重建基础兴趣场。原图尺寸必须在初始化时正确传入，才能按实际填充区域抑制黑边；底层 `I_str/I_con1/I_con2` 并非都已单独屏蔽，最终候选场仍需执行有效视域抑制。

## 3. Controller 主要方法

### 3.1 窗口与特征写入

| 方法与位置 | 作用 |
| --- | --- |
| `_extract_and_match_local_features()`，[L2141](graph_memorypool_newoptimizer.py#L2141) | 在当前注视点构造带通道门控的描述子，经 `_match_or_create_descriptor()`（[L2105](graph_memorypool_newoptimizer.py#L2105)）匹配或创建 GmemI 节点。该步骤在所有行为状态下执行。 |
| `_select_dynamic_window()`，[L1680](graph_memorypool_newoptimizer.py#L1680) | 构造工作响应和屏障场，在多个面积预算下生长候选窗口，评分后选优。平滑等效半径，返回 `WindowState` 并更新控制器窗口字段。 |
| `_seeded_grow()` / `_score_window_candidate()`，[L1618](graph_memorypool_newoptimizer.py#L1618)、[L1643](graph_memorypool_newoptimizer.py#L1643) | 从工作锚点或注视点生长连通工作区域；综合响应一致性、前沿、有效面积、细节与尺度变化评价候选窗口。 |
| `_sample_window_keypoints()`，[L2204](graph_memorypool_newoptimizer.py#L2204) | 提取 `known_confirm`（已知确认）、`known_extend`（已知延伸）、`novel_seed`（新特征种子）、`event_peak`（事件峰值），经过尺度筛选、非极大值抑制及数量限制。 |
| `_write_window_keypoints()`，[L2357](graph_memorypool_newoptimizer.py#L2357) | 匹配或创建关键点对应的 GmemI 节点，必要时创建未完成的 GmemII 语义节点；即时写入锚点关系与跨步 `temporal_adjacent` 关系，更新细节统计。 |
| `_write_keypoint_relation()` / `_update_surface_extent()`，[L2333](graph_memorypool_newoptimizer.py#L2333)、[L2285](graph_memorypool_newoptimizer.py#L2285) | 将关键点相对锚点的位移编码为对数半径和角度；按角度分箱维护面特征覆盖半径，并写入 `self_extent` 自关系。 |

`WindowState` 的 `mask` 表示当前工作区域，`frontier` 表示尚可继续探索的窗口边界；`r_eff` 是平滑后的等效半径，`area` 是实际窗口面积，`beta = attention_budget / area`。当前写入尺度门限由 `_local_response_scale_min()`（[L2154](graph_memorypool_newoptimizer.py#L2154)）根据半径比计算，`beta` 尚未直接参与该门限。

### 3.2 状态竞争与执行

| 方法与位置 | 作用 |
| --- | --- |
| `_compute_state_observation()`，[L1799](graph_memorypool_newoptimizer.py#L1799) | 计算窗口内新颖度、前沿、有效视域占比、完成度、窗口内外召回、关系缺口、失败量、窗口外新颖度与恢复价值，写入 `current_state_observation`。 |
| `_score_optimizer_states()`，[L1845](graph_memorypool_newoptimizer.py#L1845) | 将观测量加权组合为四种状态得分，并为当前状态增加惯性项，写入 `current_state_scores`。 |
| `_select_behavior_state()`，[L1922](graph_memorypool_newoptimizer.py#L1922) | 先判断 REVIEW 抢占；否则取最高分状态，再应用恢复门限、最小保持年龄和切换分差限制。抢占时还会挂起工作并设置 `review_target`。 |
| `_set_behavior_state()`，[L1914](graph_memorypool_newoptimizer.py#L1914) | 提交 `behavior_state`，同步兼容字段 `state`；状态变化时清零 `state_age`，同状态提交时加一。 |
| `_suspend_current_work()` / `_restore_suspended_work()`，[L1881](graph_memorypool_newoptimizer.py#L1881)、[L1903](graph_memorypool_newoptimizer.py#L1903) | 将当前语义、锚点、掩膜和探索统计压栈或恢复。挂起后清空活动工作指针；恢复不是全部运行时矩阵的快照恢复。 |
| `_maybe_complete_work()`，[L3024](graph_memorypool_newoptimizer.py#L3024) | 完成度高且前沿、新颖度、缺口均低于各自门限时，调用 `_transition_to_macro()`（[L3604](graph_memorypool_newoptimizer.py#L3604)）完成语义、挂接 GmemIII、累计历史并清理工作上下文。 |
| `_execute_optimizer_state()`，[L3959](graph_memorypool_newoptimizer.py#L3959) | 根据 `behavior_state` 分派到下表四个独立处理方法。 |

| 行为状态 | 执行方法与位置 | 本步行为及可能的直接切换 |
| --- | --- | --- |
| `MACRO_SEARCH` | `_run_macro_search()`，[L3844](graph_memorypool_newoptimizer.py#L3844) | 在全局候选场中选点。调用 `_decide_next_saccade_macro()`（[L3423](graph_memorypool_newoptimizer.py#L3423)），综合面连续性、语义与历史抑制、低通支持及距离惩罚。 |
| `MICRO_BUILD` | `_run_micro_build()`，[L3924](graph_memorypool_newoptimizer.py#L3924) | 写入当前点及窗口关键点，补投影新写入节点；检查完成条件，否则计算局部向量并移动。完成时可在本步切换到 MACRO 并执行全局选点。 |
| `REVIEW_VERIFY` | `_run_review_verify()`，[L3868](graph_memorypool_newoptimizer.py#L3868) | 有远处目标时先移动，后续执行时再调用 `_bfs_recognize()` 验证；成功则激活语义、添加期望并转恢复或 MACRO；失败则根据局部证据、恢复价值转 MICRO、恢复或 MACRO。 |
| `SUSPEND_RESUME` | `_run_suspend_resume()`，[L3914](graph_memorypool_newoptimizer.py#L3914) | 弹出挂起工作并回到锚点，转 MICRO；无可恢复工作则转 MACRO。 |

### 3.3 局部眼动向量

| 方法与位置 | 作用 |
| --- | --- |
| `_update_dynamic_potential()`，[L2868](graph_memorypool_newoptimizer.py#L2868) | 衰减旧 `dynamic_matrix`，在窗口扩张区域内叠加边缘、面、事件和关系缺口势，减去回访抑制，截断到 `[0, 1]`；区域外仅衰减。 |
| `_butterfly_potential()` / `_surface_potential()`，[L2757](graph_memorypool_newoptimizer.py#L2757)、[L2799](graph_memorypool_newoptimizer.py#L2799) | 分别形成沿边缘切向延伸的蝶形势，以及面区域外壳与前沿势。 |
| `_event_potential()` / `_gap_potential()`，[L2825](graph_memorypool_newoptimizer.py#L2825)、[L2839](graph_memorypool_newoptimizer.py#L2839) | 分别形成局部事件势及已有关系预测中的缺口势。 |
| `_compose_local_vector()`，[L2928](graph_memorypool_newoptimizer.py#L2928) | 由局部候选场的加权位移计算方向，合成势场、低置信 Gpos、缺口、新颖性、上一方向惯性，并减去回访方向分量。 |
| `_decide_next_saccade_micro_vector()`，[L2963](graph_memorypool_newoptimizer.py#L2963) | 标记已访问区域，更新势场并合成向量；模长不足时停留并增加失败量，否则限制步长、裁剪有效位置并更新方向记忆。返回 `VectorDecision`。 |
| `_jump_fixation()`，[L3840](graph_memorypool_newoptimizer.py#L3840) | 保存上一注视点并提交有效视域内的新注视点，是各状态共同的眼动落点更新入口。 |

MICRO 的落点满足 `p_next = clamp_valid(p + ClipStep(v))`，步长上下限随 `r_eff` 调整。向量模长不足时，本步不随机移动；增加的失败量在下一步观测和状态竞争中生效。

## 4. 主要状态变量及变更位置

运行时变量集中初始化于 `Controller.__init__()`（[L1425](graph_memorypool_newoptimizer.py#L1425)）。它们应与 `MemoryConfig` 中固定的调参项区分。

| 变量 | 含义 | 重要变更方法与位置 |
| --- | --- | --- |
| `behavior_state`、`state`、`state_age` | 新行为状态、兼容主阶段、同状态提交计数 | `_set_behavior_state()` L1914；由 `run_step()` 和各状态处理方法调用。`_transition_to_macro()` L3604 还会直接修改 `state`，新完成路径随后同步新行为状态。 |
| `current_step` | 非空输入的累计运行步数，参与挂起工作老化 | `run_step()` L3975 递增；`_resume_value()` L1789 读取。 |
| `fixation_point`、`prev_fixation_point` | 当前及上一注视点 | `_jump_fixation()` L3840；`run_step()` L3975 开始时还会裁剪当前点。 |
| `active_semantic_id`、`anchor_position`、`semantic_mask` | 当前工作语义及空间上下文 | `_ensure_work_semantic_from_keypoints()` L2262、`_ensure_work_semantic_from_local_nodes()` L2988 创建；挂起/恢复 L1881/L1903；REVIEW 激活 L3850、处理 L3868；完成清理 L3604。MACRO L3844 也会改写锚点。 |
| `suspend_stack`、`review_target` | 待恢复任务与 REVIEW 目标位置 | 抢占选择 L1922、挂起 L1881、恢复 L1903；REVIEW L3868 清空目标。 |
| `work_window/response/barrier/frontier`、`work_radius_eff`、`current_window_state` | 当前窗口、响应、屏障、前沿及尺度 | `_select_dynamic_window()` L1680 每步更新；`_ensure_runtime_maps()` L2434 初始化或按尺寸重建窗口。 |
| `current_state_observation`、`current_state_scores` | 状态竞争输入及得分 | `_compute_state_observation()` L1799、`_score_optimizer_states()` L1845，每步在状态执行前计算。 |
| `fail_energy`、`prev_local_vector` | 局部运动失败积累与上一成功方向 | `_decide_next_saccade_micro_vector()` L2963：弱向量增加失败量；有效向量衰减失败量并更新单位方向。普通状态切换不自动清零。 |
| `review_recent_fail` | 近期验证失败抑制 | `run_step()` L3975 每步衰减；`_run_review_verify()` L3868 成功清零、失败时再次衰减并累加。 |
| `dynamic_matrix`、`m_aversion` | 短期局部势与访问抑制 | 势更新 L2868；`_mark_micro_fixation()` L3247 累加访问抑制；初始化 L2434，完成清理 L3604 会重置访问抑制，但不清空动态势。 |
| `last_keypoints`、`prev_keypoints` | 本步候选及跨步关系连接所用关键点 | 采样 L2204、写入 L2357；挂起 L1881 及部分 REVIEW 结束路径 L3868 清空跨步关键点。 |
| `micro_explore_time`、`micro_max_span` | 工作探索步数与距锚点最大跨度 | MICRO L3924 累计；挂起/恢复 L1881/L1903 保存与恢复；完成 L3604 归档并清零。 |
| `semantic_history`、`active_entity_id` | 已处理区域历史与 GmemIII 实体编号 | `_transition_to_macro()` L3604 累加区域历史，必要时创建实体并挂接当前语义。 |
| `E_input_gmem_*_acc` | 三层节点本步的激活输入 | 局部提取 L3975、语义重叠 L3811、关键点写入 L3924、REVIEW 激活 L3850 累积；`_tick_all_nodes()` L3824 消费后清空。 |

`NeuronState`（静息、激活、不应期）是记忆节点自身的状态，与控制器行为状态不同。其 `activation_level/state_flag/timer` 由各节点的 `tick_update()` 更新（[L552](graph_memorypool_newoptimizer.py#L552)、[L623](graph_memorypool_newoptimizer.py#L623)、[L658](graph_memorypool_newoptimizer.py#L658)），统一调用处为 `_tick_all_nodes()`。

## 5. 主要配置参数

| 参数组及定义位置 | 对行为的影响 | 主要读取方法 |
| --- | --- | --- |
| `state_bias_*`、`state_inertia_*`、`z_*`，[L303](graph_memorypool_newoptimizer.py#L303) | 设置各状态基准倾向、维持惯性及观测权重。MICRO 偏好新颖度、前沿、缺口与有效区域；MACRO 偏好失败、低质量、完成和远处新颖性；REVIEW 偏好召回；恢复偏好剩余工作价值。 | `_score_optimizer_states()` L1845 |
| `state_min_age=2`、`state_switch_margin=0.20`，[L288](graph_memorypool_newoptimizer.py#L288) | 普通竞争切换的最小状态年龄与分差；不约束 REVIEW 抢占或处理方法内的直接切换。 | `_select_behavior_state()` L1922 |
| `theta_review_preempt=0.45`、`review_preempt_margin=0.15`，[L290](graph_memorypool_newoptimizer.py#L290) | 窗口外召回及 REVIEW 相对 MICRO/MACRO 得分优势均满足门限时抢占。 | `_select_behavior_state()` L1922 |
| `resume_value_threshold=0.25`、`review_recent_decay=0.90`，[L292](graph_memorypool_newoptimizer.py#L292) | 控制恢复资格和近期验证失败的衰减。恢复价值结合剩余完成度、前沿、缺口与挂起时长。 | `_resume_value()` L1789、状态选择 L1922、REVIEW L3868、`run_step()` L3975 |
| `theta_vector_norm=1.5`、`fail_decay=0.85`、`fail_increment=1.0`，[L294](graph_memorypool_newoptimizer.py#L294) | 控制弱向量判定和失败积累，间接提高后续 MACRO 倾向；`fail_increment` 也用于 REVIEW 失败。 | 局部向量决策 L2963、REVIEW L3868 |
| `theta_complete=0.72`、`theta_frontier_close=0.05`、`theta_novel_close=0.05`、`theta_gap_close=0.08`，[L297](graph_memorypool_newoptimizer.py#L297) | 必须同时满足的工作完成门限。完成度须大于门限，其余三项须小于门限。 | `_maybe_complete_work()` L3024 |
| `window_r_*`、`window_candidate_count`、`window_seed_radius`、`window_grow_steps`、`theta_window_*`、`theta_barrier`、半径平滑参数，[L270](graph_memorypool_newoptimizer.py#L270) | 控制窗口候选范围、生长限制、响应门控及尺度变化速度，进而影响状态观测与眼动步长。 | 窗口构造 L1573 至 L1680 |
| `kappa_step_min/max`、`kappa_edge_step`、`kappa_surface_delta`，[L281](graph_memorypool_newoptimizer.py#L281) | 将窗口尺度转换为步长限制、边缘推进尺度及面外壳尺度。 | 步长限制 L2655、边缘势 L2757、面势 L2799、局部势更新 L2868 |
| `dynamic_decay`、`dynamic_w_*`、`vector_w_*`，[L327](graph_memorypool_newoptimizer.py#L327) | 分别控制势场记忆、各势分量及最终运动方向分量的权重。 | 势更新 L2868、向量合成 L2928 |
| `theta_edge_*`、`theta_surf_*`、`theta_event`、`theta_novel`、`keypoint_nms_radius`、`max_keypoints_dim*`、`ell_*`，[L333](graph_memorypool_newoptimizer.py#L333) | 控制响应分段、事件/新特征采样、空间去重及写入尺度；`theta_novel` 还用于 REVIEW 失败后的学习判断。 | 关键点采样 L2204、局部势 L2757、REVIEW L3868 |
| `interest_margin_radius=24`、`gamma_macro_dist=0.001`，[L186](graph_memorypool_newoptimizer.py#L186)、[L223](graph_memorypool_newoptimizer.py#L223) | 有效视域边缘收缩与 MACRO 距离惩罚。 | 有效矩形 L1217、MACRO 选点 L3423 |

`state_temperature` 当前只有定义，状态选择采用确定性最高分及滞回判断，未进行温度采样。`attention_budget` 当前用于记录窗口 `beta`，不能直接理解为已完整控制关键点写入预算。

## 6. 调试时的读取顺序

优先检查 `behavior_state`、`current_state_scores`、`current_state_observation`，再检查 `work_window`、`dynamic_matrix`、`fail_energy` 和 `review_target`。局部向量分解位于 `debug_optimizer['vector_components']`，模长位于 `debug_optimizer['vector_norm']`。

`debug_optimizer['state']` 是执行状态处理方法之前的快照；处理方法可能再次切换状态，因此本步结束状态应读取 `behavior_state`。同理，本步关键点写入、失败累加后的结果，不会自动回填到已经计算好的观测量与得分。`state_age` 按 `_set_behavior_state()` 的调用更新，不是严格的物理帧计数。

`current_interest_map` 随执行状态表示不同对象：MACRO 候选场、MICRO 动态势、REVIEW 召回场或恢复掩膜。`matrix1/2/3/4` 分别用于动态势、工作窗口、前沿和 MACRO 语义响应叠加调试；未经过相应执行路径时可能保留旧值。

当前 `_semantic_recall_map()`（[L1741](graph_memorypool_newoptimizer.py#L1741)）以语义关联节点的 GposI 响应均值形成召回代理，并非调用 GposII 结构合成。新执行链尚未调用 `l2_structure_synthesis()` 或独立 GposIII 响应，也未消费 `I_exp` 来引导眼动。旧 `MainPhase`、`MicroIntention` 和 `_process_phase_observation()` / `_decide_next_by_phase()` 保留在文件中，但新 `run_step()` 的行为分派依据是 `OptimizerState`。
