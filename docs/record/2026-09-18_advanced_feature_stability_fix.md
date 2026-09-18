# 高级模态数值稳定性、方向置信度与分区入口修复

日期：2026-09-18。依据 [输入诊断](2026-09-18_advanced_modality_input_diagnosis.md)，修改固定特征计算及其输入合同；没有执行长期数据集训练或调整分割阈值寻优。

## 1. 为什么不只在最终输出上调用 nan_to_num

aps 的局部非法平方根会经过整图最大值归一化变成整张 NaN。最终才清洗只能得到全零尺度，无法恢复原特征。本次采用“先修复公式，局部清洗，再做空间归一化”：`nan_to_num(nan=0, posinf=0, neginf=0)` 明确将异常归零，不使用默认的极大有限值替代 Inf；无效点的方向置信度为零，不将其零角度解释为可靠方向。原始外部非有限输入在图模块仍经过有效性 mask 检查。

CNN helpers 会把 float16/bfloat16 特征运算提升为 float32；本次主要验证 float32 前向，未声明端到端可微性或反向梯度数值验证。

## 2. 特征计算修复

修改 [features.py](../../src/nns/cnns/features.py)：

- curv：使用 hypot 计算 Hessian 分量范数，去掉平方根内导致无结构区正偏置的 epsilon；零导数输出零，再用带下限的整图最大值归一化。极小量与非有限输入不会将全图污染为 NaN。
- aps：先按局部结构张量量级缩放，用 `hypot(Jxx-Jyy, 2Jxy)` 计算特征值间距；将理论非负的特征值限制为非负。使用 dtype 相对正则项与对数差计算长短轴比，避免负数开方和比值溢出。零能量或各向同性位置输出零，删除 `log(ratio + 1e-6)` 在零结构区引入的偏置。
- ori：根据结构能量与各向异性产生置信度。结构能量下限通过 `MultiScaleFeatureBank(..., energy_floor=1e-8)` 显式提供；无能量或无明确主轴的位置置信度为零。
- 方向上采样：插值 `confidence*cos(2θ)` 与 `confidence*sin(2θ)`，再恢复轴向角度，插值合向量幅度作为置信度。正负角度端点不再被错误平均为垂直方向；相互抵消的方向产生低置信度。
- 独立导数提取器及复用梯度后的二阶导数采用 replicate 边界，避免常量输入在这些步骤被零填充制造边缘。上游 Retina 自身的梯度边界行为未整体改写；图片有效掩码仍需保留。
- 小图片的池化尺度限制在实际 H/W 内；显式验证尺度和能量下限参数。

`MultiScaleFeatureBank.forward` 默认仍返回 `(curvature, aspect, orientation)`。新参数 `return_confidence=True` 额外返回 `[B,S,H,W]` 方向置信度，供新图输入合同使用。

## 3. 方向输入合同与写入一致性

修改 [graph_memorypool_onceoptimizer.py](../../src/nns/memorygraphs/graph_memorypool_onceoptimizer.py)：

1. 新增 `configure_orientation_contract(cfg, S)`，显式将整数模态 4 定义为 `[S 个弧度角度通道, S 个置信度通道]`。Notebook 默认 S=3，所以 ori 有 6 个通道。名称输入的方向仍为 CNN 的 `theta/(pi/2)`，进入图时仅转换角度，置信度保持 [0,1]。
2. 置信度通道是 `gate_only`，不会进入周期角度相似度；原型 mask 与 feature contract 同步声明。名称输入必须同时提供形状匹配的 `orientation_confidence`，通道数量错误直接报错。
3. `tau_ori_confidence=.05` 为显式门控配置。当前完整多尺度描述子要求所有角度尺度均有超过该阈值的置信度，再结合原有局部方向一致性门控。暂未实现逐像素部分尺度 mask，因此采用保守的全尺度有效策略；遮挡或尺度冲突时可能减少覆盖，不能用未通过门控证明物体不存在。
4. 方向一致性与跨尺度轴向聚合按置信度加权。ori 的 Trace_1D 沿轮廓切向连接，因此使用梯度主轴的垂直方向；模板存储的角度仍是梯度法线，没有混淆描述符角度和分区切向。
5. 不使用新合同时，整数角度及三输出 bank API 仍可调用；旧角度输入没有能量信息，无法仅从零角度判断是否是平坦区。需要启用新合同才能获得置信度门控。

curv、aps、ori feature_specs 增加 `encoding_version='stable_structure_v2'`，新方向合同还记录布局与角度单位。已有 `load_state_dict` 对 feature contract 的检查会拒绝旧记忆，避免悄悄混用不同定义；本轮必须重新建图，不自动迁移旧原型。

## 4. 分区前处理与 Notebook

curv 的局部二阶矩方向估计仅使用可写区域内的强度，减少区域外信号对切向的污染。种子 max-pooling 也仅在当前 active 支持位置上寻找极大值，不允许不可写位置压制真实候选。没有降低 `.6` 种子阈值来制造种子。

[Notebook](../../src/test_memorypool_onceoptimizer.ipynb) 默认请求方向置信度，并在 `make_config` 中设置对应合同。新增原始 CNN 各模态、各尺度的 NaN/Inf 计数和有限值范围，避免清洗后的 Reliability 图掩盖输入异常。保留原有支持/种子/区域图，说明新方向格式及重新建图要求。

## 5. 验证结果

18 项小规模 CPU 回归通过，覆盖先前的粗筛/匹配测试，以及新增零图、常量图、随机秩一结构张量、局部 NaN/Inf、不连续角度端点、方向置信度门控、置信度不参与相似度、切向方向、旧合同拒绝及小尺寸图像。修改后的 Python 文件与全部 Notebook 代码单元静态编译通过。

另外使用 Notebook 指定的 `ILSVRC2014_train_00060030.JPEG`，分别在 CPU 和实际 CUDA 环境执行前向与一次性构图；没有执行长期训练。当前沙箱不可访问 CUDA，因此 CUDA 单图验证通过获准的沙箱外运行完成，临时移除冲突的 `LD_LIBRARY_PATH`，未修改系统环境。

| 指标 | 修复前 CPU | 修复后 CPU | 修复后 CUDA |
| --- | ---: | ---: | ---: |
| aps 尺度 1 NaN 像素 | 65,536 | 0 | 0 |
| curv 初始种子 | 0 | 76 | 76 |
| aps 支持像素 | 0 | 21,913 | 21,912 |
| aps 区域 / 节点 | 0 / 0 | 51 / 385 | 51 / 383 |
| ori 区域 / 节点 | 102 / 1,024 | 51 / 282 | 51 / 282 |

修复后所有 curv/aps/ori 尺度及方向置信度均无 NaN/Inf；独立 bank 的零图和常量图测试三种模态输出均为零。CPU/CUDA 的少量节点差异来自连续分数及后续离散选择的数值敏感性，本次不宣称逐位一致。

ori 区域数下降还受到 aps 恢复参与全局区域预算分配的影响，不能把下降全归因于方向修复。curv 仍有约 75% 支持点被判断为方向不可靠事件，aps 最细尺度的秩一张量也使各向异性较易饱和；这些是后续表征与分割质量评估的问题。当前结果证明数值异常已消除、五模态路径可运行，不等于自然图像分割质量或长期判别能力已经验证。

复现脚本及 CPU/CUDA 统计见 [results/modalities_fix_20260918](../../results/modalities_fix_20260918/)。需重启 Notebook 内核，从头执行编码、配置和建图单元，不复用旧 pools。
