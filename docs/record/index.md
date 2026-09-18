本文档是用于查找代码更改日志的索引文件

| 文件名 | 更改概括 | 提交者 | 提交时间 |
| --- | --- | --- | --- |
| 2026-09-18_supervised_graph_learning_plan.md | 核对 1518 个目标框，设计标注确认实体、框监督学习、增长图分类读出及无框检测/视频扩展完整方案 | chatgpt | 2026-09-18 |
| 2026-09-18_index_parallelism_growing_graph_classifier.md | 复查 38 图运行、42.53% 粗筛与关系瓶颈；审查自适应编码树、多核方案及增长图分类头生命周期 | chatgpt | 2026-09-18 |
| 2026-09-18_grad_bound_edge_attributes.md | aps/ori 默认继承 grad 分区、邻接与有效锚点；增加父区域图示和分区合同，21 项回归及真实单图构图通过 | chatgpt | 2026-09-18 |
| 2026-09-18_advanced_feature_stability_fix.md | 修复 aps NaN 与零输入偏置；引入方向置信度/轴向插值/切向合同，修复种子掩码；18 项回归及 CPU/CUDA 单图验证通过 | chatgpt | 2026-09-18 |
| 2026-09-18_advanced_modality_input_diagnosis.md | 真实 CNN 单图复现 aps 首尺度全 NaN，定位特征值相消与归一化传播；审查 curv/ori 分区及零输入退化 | chatgpt | 2026-09-18 |
| 2026-09-18_binary_coarse_multimodal_implementation.md | 接入保守区间位图粗筛与模板缓存；默认启用五模态、统一预算及投影权重，补充阈值诊断和 13 项回归 | chatgpt | 2026-09-18 |
| 2026-09-18_binary_coarse_filter_review.md | 审查现有分层位图/倒排机制，区分近似召回与安全排除，推导加权覆盖必要条件及接入方案 | chatgpt | 2026-09-18 |
| 2026-09-17_onceoptimizer_serial_bottleneck_discrimination_review.md | 新增第 0 章审查 region_matching 实际算法；复查 29 图单核瓶颈、后续优化及判别模块需求 | chatgpt | 2026-09-17 |
| 2026-09-17_onceoptimizer_progressive_matching_implementation.md | 实现可评估性前移、逐批上界早退和局部响应；预留默认关闭的预算/未提交接口；CPU/GPU 回归与真实区域对照通过 | chatgpt | 2026-09-17 |
| 2026-09-17_onceoptimizer_algorithm_revision_proposal.md | 评估不改双塔层级的 one_step 调度、区域候选搜索、保守上界早退与预算延期学习，区分等价加速和行为变更 | chatgpt | 2026-09-17 |
| 2026-09-17_onceoptimizer_postoptimization_gpu_plan.md | 复查优化后 10 图日志，定位 256 MiB 缓存频繁淘汰及后置筛选，评估显存容量与 GPU 批量匹配迁移方向 | chatgpt | 2026-09-17 |
| 2026-09-17_onceoptimizer_response_cache_performance.md | 共享有界响应缓存、批量 GPU 回传及必要条件筛选；4 项回归通过，真实图查询/局部匹配对照一致 | chatgpt | 2026-09-17 |
| 2026-09-15_onceoptimizer_performance_cuda_report.md | 分析 CPU 与 6 图 GPU 实测；补充低占用、98.9% 检索耗时与密集回退证据及优化优先级；记录 CUDA 迁移 | chatgpt | 2026-09-15 |
| 2026-09-15_onceoptimizer_dataset_reliability_resources.md | 扩展第 8 节为 659 图训练／留出与复查测试，记录证据、平移一致性、内存／显存及增量日志；仅静态编译 | chatgpt | 2026-09-15 |
| 2026-09-12_hierarchy_retrieval_consolidation_implementation.md | 实现平移模式三层超图检索、位图反向关联、跨观察证据与实体巩固，扩充 Notebook 调试图；仅编译和静态检查 | chatgpt | 2026-09-12 15:48:12 CST |
| 2026-09-12_hypergraph_similarity_index_design.md | 澄清层级超图与模板等效条件，补充三层相似度、二进制关联索引、稀疏传播及复杂度边界 | chatgpt | 2026-09-12 |
| 2026-09-11_graph_and_optimizer_design.md | 审查三层 Gmem/Gpos，完善结构检索与虚拟统一锚点，设计 GmemIII 跨观察归并和关系巩固及缺失模块清单 | chatgpt | 2026-09-11 23:10:00 CST |
| 2026-09-11_onceoptimizer_notebook_plot_annotations.md | 为一次性构图 Notebook 的分割、采样、关系图和检索图增加读图说明、标记图例与色条，保留已有测试输出 | chatgpt | 2026-09-11 21:26:05 CST |
| 2026-09-11_graph_memorypool_onceoptimizer_implementation.md | 续写一次观察区域构图器，保留 Gmem/Gpos 与 run_step 接口，新增区域采样、星型及接触写入和调试 Notebook；仅编译检查 | chatgpt | 2026-09-11 20:27:36 CST |
| 2026-09-11_alg_onceoptimizer_beta.md | 新建一次性区域构图器设计：同模态连续分区、实例采样、区域星型与稀疏跨区接触，明确现有 Gmem/Gpos 兼容边界及验证计划 | chatgpt | 2026-09-11 15:08:41 CST |
| 2026-09-09_newoptimizer_saccade_parameter_diagnosis.md | 为新优化器增加执行状态诊断，重构调参 Notebook，实测区分 REVIEW 往返、MACRO 停留与 MICRO 局部振荡 | chatgpt | 2026-09-09 15:35:00 CST |
| 2026-09-07_graph_memorypool_newoptimizer_vector_state.md | 在 graph_memorypool_newoptimizer 中实现 Alg_optimizer_beta 视窗口向量优化器、动态状态竞争与窗口关键点即时写入 | chatgpt | 2026-09-07 22:24:34 CST |
| 2026-07-26_alg_optimizer_beta_gpos_workspace_surface_self_relation.md | 修订 Alg_optimizer_beta：用 Gpos 响应域生成不规则工作视窗，并新增二维 surface 面域自指关系与外沿壳层引导 | chatgpt | 2026-07-26 15:26:22 CST |
| 2026-07-23_alg_optimizer_beta_dynamic_window_scale.md | 在 Alg_optimizer_beta 中新增动态视窗口尺度机制、固定注意力预算下的最低写入尺度和尺度竞争流程 | chatgpt | 2026-07-23 22:23:46 CST |
| 2026-07-23_alg_optimizer_beta_local_vector_route.md | 新建 Alg_optimizer_beta，讨论 MICRO 从全局 drive 改为视窗口向量引导，并补充动态竞争状态机与窗口写入流程 | chatgpt | 2026-07-23 19:50:38 CST |
| 2026-07-21_alg_beta_unified_micro_drive_review.md | 在 Alg_beta 过渡带统一图表征章末新增通用 MICRO drive 评判、支持维度归一化与工作关系边增强方案 | chatgpt | 2026-07-21 19:53:55 CST |
| 2026-07-20_representation_graph_consciousness_feasibility.md | 追加 RepresentationLearn 中图结构视觉表征、再现与联想功能的可行性分析和验证标准 | chatgpt | 2026-07-20 22:34:04 CST |
| 2026-07-20_alg_beta_transition_feature_graph_revision.md | 覆盖 Alg_beta 渐变边界章节，改为以 RGB_TRANSITION 低层特征族和统一 MICRO drive 表征过渡带 | chatgpt | 2026-07-20 22:07:32 CST |
| 2026-07-20_alg_beta_gradient_boundary_frontier_learning.md | 追加 Alg_beta 渐变边界、语义前沿蚕食、动态漂移证据与 MICRO 意图扩展方案 | chatgpt | 2026-07-20 20:04:13 CST |
| 2026-07-20_graph_memorypool_valid_view_drive_suppression.md | 修正 graph_memorypool 中黑边伪边缘对各阶段 drive 与兜底眼跳选点的干扰 | chatgpt | 2026-07-20 15:58:16 CST |
| 2026-07-18_graph_memorypool_channel_contract_schema.md | 实现 graph_memorypool 通道契约、门控/相似度掩码分离、分组相似度与 Gpos 门控投影 | chatgpt | 2026-07-18 22:06:15 CST |
| 2026-07-18_alg_beta_universal_extraction_revision.md | 续写 Alg_beta 探讨二修订版，统一五类输入的通道依赖、门控掩码与分组相似度检索规范 | chatgpt | 2026-07-18 21:43:20 CST |
| 2026-07-17_graph_memorypool_boundary_trace_hysteresis.md | 修正 graph_memorypool 中 BOUNDARY_SEEK 落点确认、CONTOUR_TRACE 迟滞退出与边界访问延迟抑制 | chatgpt | 2026-07-17 22:14:01 CST |
| 2026-07-17_alg_beta_micro_trace_debug_adjustment.md | 补充 Alg_beta 中 BOUNDARY_SEEK 死锁、CONTOUR_TRACE 早退与局部特征过度合并的算法修正建议 | chatgpt | 2026-07-17 16:08:15 CST |
| 2026-07-16_graph_memorypool_micro_hyperedge_refactor.md | 重构 graph_memorypool：新增 MICRO 意图状态机、多重关系边与超边式 SemanticNode，并缩短 run_step | chatgpt | 2026-07-16 21:16:05 CST |
| 2026-07-16_alg_beta_hyperedge_section_restructure.md | 重排 Alg_beta 眼动引导章节，并在自指边后补入多重边/超边规范 | chatgpt | 2026-07-16 20:45:06 CST |
| 2026-07-16_learn_micro_gpos_target_fields.md | 将 Alg_beta 中 LEARN_MICRO 各拓扑意图目标场改写为基于 Gpos 响应的计算式 | chatgpt | 2026-07-16 20:20:20 CST |
| 2026-07-12_gpos_position_similarity_review.md | 补充 Gpos 位置相似度检索审查、图边多重性规范与拓扑意图场的 Gpos 响应依赖 | chatgpt | 2026-07-12 21:31:44 CST |
