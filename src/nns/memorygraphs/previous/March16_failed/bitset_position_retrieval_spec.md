# Bitset 精筛 + 位置精排 检索 规格文档

**版本**：v1.0  
**作者**：（由系统生成）  
**目标读者**：工程实现者、研究者、系统架构师

---

## 1 概要
本规格定义一种两阶段视觉检索索引与匹配方案：

- **一级（粗筛 / bitset coarse）**：使用按模态分段的 multi-hot bitset 做极快速候选召回；位向量表示每个节点在若干模态（edge_proto、hue_proto、curvature_bank、aspect_bank、orient_bank）上的离散类别存在性。
- **二级（精排 / position fine）**：对一级候选使用更细粒度的空间编码（Proto×Spatial-bin 二值矩阵、多尺度 grid、proto-centroid、proto-pair 相对关系）进行打分与排序，得到最终结果。

设计目标：兼顾检索吞吐（大批量对比）、检索精度（位置与结构一致性）、存储可控性与解释性。

---

## 2 总体架构

1. **离线/增量构建阶段**：从视觉流或图片中提取 proto / bank-region → 量化到类型（模态类型） → 生成 node 的 bitset 与位置编码 → 写入索引（bitset 主索引 + 倒排表 + 辅助位置表）。
2. **在线查询阶段**：对 query 计算 bitset 与位置编码→ 一级粗筛（按位相似度/加权 Jaccard）→ 取 Top-K_fast → 二级精排（centroid kernel + pairwise consistency + weighted score）→ 返回 Top-K_final。

---

## 3 数据布局与字段定义（磁盘 / 内存表示）

### 3.1 全局参数（默认）
- `P_global`：全局 proto 词表大小 = **128**（可配置）
- `P_local`：每 node 激活 proto 上限（top-P_local）= **64**（可配置）
- `scales`：multi-scale grids = **[1×1, 2×2, 4×4]** → `S_total = 21`
- `D = P_local * S_total = 64 * 21 = 1344` bits（主签名长度）
- `bitset_packed_blocks`：`ceil(D / 64) = 21` uint64 blocks
- `K_fast`（粗筛候选数）= **200**
- `K_final`（返回 top）= **10**
- `M_centroid`（精排使用 proto 上限）= **6**
- `K_pairs`（每 node 保存的 top pair count）= **12**

> 注：以上为起步配置，需通过离线 A/B 调参和规模基准测试调整。


### 3.2 Node 主记录（内存映射/持久化结构）
- `node_id` (uint32)
- `bitset_blocks` (uint64[bitset_packed_blocks])  — 主 multi-hot bitset（proto×spatial multi-scale 展平并按 proto block 分段）
- `proto_list` (uint16[P_local]) — 激活 proto id 列表（按强度排序）；实际数量用 `proto_count` 指示
- `proto_cell_masks` (uint8[P_local]) — 对应每个 proto 在细网格（例如 4×4）上的 cell bitmask（或 per-scale bitmask）
- `proto_centroids` (uint16[2*M_centroid]) — top-M_centroid 的 (x,y) 归一化定点表示（0..65535）
- `proto_vars` (uint16[M_centroid]) — 对应方差或半径（定点化）
- `pair_tokens` (uint16[K_pairs]) — top-K_pairs 的 proto-pair token（packed angle/dist bins）
- `strength` (float) — 节点强度/置信度（用于 prune & ranking）
- `last_seen` (uint32 timestamp)

> 存储建议：主记录建议存为固定长度二进制结构或 mmap-able 文件；大规模数据可把大数组移到 LMDB / RocksDB 并在内存保留索引头。

### 3.3 倒排索引（posting lists）
- `(modality, type)` -> posting list of (node_id, freq_or_strength)
- `(proto_id, spatial_bin)` -> posting list (用于更细粒度快速过滤)
- `pair_token` -> posting list (用于 pairwise 精筛)

> posting list 建议按 node strength 排序并支持 delta 编/压缩；支持并行读取和截断（只读 top-N_posting）。

---

## 4 Bitset 设计（模态分段 / multi-hot）

### 4.1 模态与位段分配（示例）
- `edge_proto`：8 bits (types: 尖锐、曲折、直线段、圆弧等共 8 类)
- `hue_proto`：6 bits (颜色桶，如 红、橙、黄、绿、蓝、紫)
- `curvature_bank`：4 bits
- `aspect_bank`：4 bits
- `orient_bank`：4 bits

**合计**：26 bits（仅示例）。在实际系统中应将模态分段扩展为 `proto×spatial` 主签名（见 1）以编码相对位置。上述小位段可作为额外语义面（存在/confidence counters）。

### 4.2 主签名（proto × spatial multi-scale）
- 选用 P_local（例如 64）proto slots 与 multi-scale S_total bins（例如 21）→ D = 1344 bits
- 布局：第 0..(P_local-1) block 每个 block 含 S_total bits，对应该 proto 在每个空间单元的存在性
- 打包为 uint64 blocks（21 个）以便高效位运算（AND/POPCNT）

### 4.3 语义位的置信度
- 不建议使用纯 0/1；可为每个 `modality-type` 预留 2-bit 计数表示置信度（0/1/2/3+），以降低单帧噪声导致的位翻转影响。

---

## 5 位置编码（精排）详细设计
位置编码由三部分组成（按精细度从低到高）：

### 5.1 Proto × Spatial-bin 二值矩阵（主位置签名） — 一级精排/主索引用
- 对 Node：对于 node 的 top-P_local proto，每个 proto 在 multi-scale grid 上做量化并把相应 bin 置 1
- Grid scales：{1×1, 2×2, 4×4}（可替换/添加 log-polar scale）
- 展平顺序：按 proto 顺序拼接各 scale 的 bin bits → 得到长度 D 的二值向量
- 支持操作：位与 + popcount（得到 intersection）和预先计算的 per-node bitcount（用于 union）→ 批量 Jaccard 计算

**表征优点**：批量化、可用向量内积/位运算做高速筛选

### 5.2 Proto centroid + variance（细粒度精排）
- 对 node 的 top-M_centroid proto 保存 (x,y) 归一化坐标（uint16）与 variance（uint16）
- 查询时计算 Mahalanobis kernel：
  \( s_i = \exp\left(-\frac{||x_q-x_n||^2}{2(\sigma_q^2+\sigma_n^2)}\right) \)
- 对 top-M_centroid 求平均或加权和作为位置相似度

**用途**：对 top-K_fast 做精排时使用，解决 bin 量化带来的近邻误差

### 5.3 Proto-pair 相对关系稀疏编码（结构信息）
- 对 node 的 top-M_local proto 计算 pairwise relative vector (dx,dy)
- 量化为 (angle_bin, log_dist_bin)，pack 为 pair_token
- 只存 top-K_pairs（按频率或置信度）作为稀疏表
- 精排时计算 pairwise 一致性得分（共享 token / angle/dist 相近数量化得分）

**用途**：区分结构/拓扑不同但组成相似的节点

### 5.4 旋转/尺度鲁棒策略
- 可选 PCA 主轴对齐（对称对象需谨慎）
- 或使用 log-polar bin 作为额外 scale，增强尺度不变性

---

## 6 检索流水线（详细）

### 6.1 查询预处理
1. 对 query image / region 提取 proto 列表与 bank 响应
2. 计算查询 `bitset`（主签名 D bits，和语义置信位）
3. 构建查询的 `proto_centroids` 与 `pair_tokens`（top M, top K）

### 6.2 一级粗筛（Bitset Coarse）
1. 批量计算 intersection counts：对节点主签名矩阵 \(X\in\{0,1\}^{N	imes D}\) 与查询 \(q\) 做乘积或对 packed uint64 blocks 做位与+popcount，得到 `intersect`。
2. 计算 union = count(X_row_bits) + count(q_bits) - intersect
3. 计算 `Jaccard = intersect / union`（或加权 Jaccard，模态/scale 权重可配置）
4. 取 Top-K_fast（默认 200）作为候选

> 优化：若候选集过大，可先用倒排索引 `(proto,scale_bin)` 对发布列表做交叉剪裁，再对剩余做位运算

### 6.3 二级精排（Position Fine）
对 Top-K_fast 中每个候选执行：
1. **centroid kernel**：对查询 top-M_centroid 与候选 centroids 计算核相似度，得到 `pos_score`
2. **pairwise consistency**：计算 query pair tokens 与 node pair tokens 的交集频率，得到 `pair_score`
3. **综合评分**：
   \( S = \alpha\cdot J_{spatial} + \beta\cdot pos\_score + \gamma\cdot pair\_score + \delta\cdot strength \)
   - 建议初始权重：\(\alpha=0.55,\beta=0.30,\gamma=0.10,\delta=0.05\)
4. 返回按 S 排序的 Top-K_final

### 6.4 后处理（可选）
- 若需要语言/符号对齐，可把 Top-K_final 的 node embedding 或 proto 集合与文本 embedding 做对齐/再排序（非本规格核心）

---

## 7 索引运维与版本策略

### 7.1 版本化 bitset 格式
- 预留 8–16 位作为版本/扩展位
- 当升级到更细粒度的 proto 或更多 scale 时，保留旧格式的兼容映射（在 ingest 时同时生成 v1 和 v2 bitset，或做在线迁移）

### 7.2 节点修剪策略
- 基于 `strength` 与 `last_seen` 做 LRU 与阈值剔除
- 周期性 consolidate：把高度重复/近似的 nodes 合并并重写索引

### 7.3 索引重建与在线兼容
- 支持批量重建（离线）与增量更新（写入 posting lists）
- 对倒排列表采用 append-only + background compaction

---

## 8 性能估算与硬件建议

### 8.1 存储估算（示例）
- 每 node 主签名：21 uint64 = 168 bytes
- proto_list & proto_cell_masks & centroids & pair_tokens 等约 200 bytes（取决 M,K）
- 总计约 **~400 bytes/node**（不含 DB 元数据与索引开销）

若节点数 N = 1M → 数据量 ≈ 400 MB（主表） + 倒排索引开销（视 posting list 长度），总体 < 2–3 GB 可用内存混合磁盘策略

### 8.2 查询成本估算（粗略）
- 粗筛：对 1M 节点的位与+popcount（21 blocks）约需 21M popcount 操作；按现代 CPU popcnt 性能及多线程，可在 10s–100s ms 量级完成（需优化/并行）
- 为降低延迟：使用倒排预筛或分区 + 多线程 + 缓存热节点

### 8.3 硬件建议
- CPU：带 popcnt 指令集的多核 CPU（AVX2/AVX512 可进一步加速）
- 内存：主表与热倒排部分建议驻留内存，冷数据放 LMDB/RocksDB
- 可选：GPU/TPU 用于大批量浮点精排（centroid kernel）但位运算仍在 CPU 高效

---

## 9 评价指标与试验计划

### 9.1 关键指标
- Recall@K_fast（粗筛是否保留正确候选）
- Precision@K_final（最终准确性）
- Mean Average Precision (mAP)
- Query latency (p50/p95/p99)
- Candidates per query after coarse filter
- Storage per node & total index size

### 9.2 实验步骤
1. 使用真实/合成数据构建基线索引（N从10k到1M）
2. Sweep 参数：P_local (32/64/128)、scales、K_fast (50/200/500)、M_centroid (4/6/8)
3. 对比三种 coarse 筛选策略：bitset 单次位运算、倒排交叉、混合
4. 评估对扰动（平移/旋转/尺度/遮挡）的鲁棒性

---

## 10 实现注意与扩展路径

- **并行化**：粗筛阶段高度并行；建议按节点块分片并多线程或分布式执行
- **压缩**：posting lists 使用 delta 编码，主签名可按行打包并对齐到 cache line
- **监控**：在线统计每个位/每个 proto 的命中率与误判率，供在线调参
- **扩展**：如需更强结构学习，可加训练型 learned-binary encoder 替换 bitset 子段，或在精排阶段加入图神经网络对候选做端到端评分

---

## 附录 A — 参数默认表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| P_global | 128 | 全局 proto 词表 |
| P_local | 64 | node local proto slots |
| scales | [1,2,4] | multi-scale grid |
| D | 1344 bits | 主签名位长度 |
| bitset_blocks | 21 | uint64 blocks |
| K_fast | 200 | 粗筛候选数量 |
| K_final | 10 | 返回数量 |
| M_centroid | 6 | 精排使用 proto 上限 |
| K_pairs | 12 | 存储 pair tokens 数量 |

---

## 附录 B — 检索伪代码概要

> 由于你要求不包含实现细节代码，此处仅提供算法流程伪码（高层）

1. compute_query_signature(query) -> q_bitset, q_centroids, q_pairs
2. candidates = bitset_coarse_search(q_bitset)   # top K_fast via Jaccard
3. for node in candidates: compute pos_score(node,q_centroids)
4. for node in candidates: compute pair_score(node,q_pairs)
5. score = alpha*J_spatial + beta*pos_score + gamma*pair_score + delta*node.strength
6. return top-K_final by score

---

## 结语
本规格旨在提供一套工程可落地的检索方案：用解释性强、可向量化的 bitset 做低延迟粗筛，用分层的位置编码（binary grid + centroid + pairwise）做鲁棒精排。该方案便于逐步迭代：先实现 bitset 主签名与位运算快速筛，再按需要逐步打开 centroid 与 pairwise 精排。请按你的数据规模与延迟目标做参数扫描。

如果你需要，我可以下一步生成：
- 字节级别的二进制布局表（每个字段的字节偏移）以便直接实现 mmap 结构；或
- 针对你现有 `graph_memorypool.py` 的补丁/变更说明，告诉你如何在代码中插入/映射这些字段（包含倒排 schema 与写入流程）。

---

*结束*

