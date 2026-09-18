# 标签、目标框监督与判别任务实施方案

日期：2026-09-18。完整报告与算法细节见 [Alg_SupervisedGraphLearning.md](../algorithm/Alg_SupervisedGraphLearning.md)。本次为方案设计，未修改代码或启动训练。

## 结论

标签和框监督可行，建议先完成“已知目标框分类”，再扩展“无框目标检测”。允许有效标注实例第一次出现即建立 annotation_confirmed、受保护的稳定 GmemIII，但独立视觉证据仍只计一次；框内背景成员不能因此永久固定。

完整方案覆盖标注解析、ROI 坐标变换、框/软前景约束、grad 属性组选择、监督实体与生命周期、同类一对一角色匹配、类别证据头与可训练分类头、增长节点管理、无框候选/定位/NMS、视频接口、数据划分、少样本预算和验收标准。

## 数据检查

本地 659 个 XML 中有 530 张图含目标，合计 1,518 个框、125 类；129 张没有 object 条目，不能未经核实当全负样本。只有 16 类出现在至少 10 张独立图片中，适合作为首轮有独立划分的候选类别。当前 Notebook 示例 00060030 恰好无目标条目，监督演示应换用有框样本。

## 核心实现顺序

1. 新增 annotation/ROI 数据合同及分组划分；先建立没有标签泄漏的目标裁剪流程。
2. 将框内有效区域组成目标级提案，保留 grad/aps/ori 的关联，加入受保护标注种子、来源与同 episode 幂等记录。
3. 用各类实体最大匹配证据组成固定类别维度向量，先完成无额外训练头的少样本分类基线；再用独立训练子集拟合共享读出和分类头。
4. 改善同类不同 GmemII ID 的角色对齐，允许同类多姿态模板，纠正背景成员。
5. 加入无框候选、多尺度 ROI 编码、框逆变换、去重及检测评价。
6. 后续用真实光流、相机运动补偿和跟踪生成对象观察；当前 Retina 的 flow 输出仍为零占位，不能直接用于视频前景学习。

框初始化前景提取可参考 [GrabCut 原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf)；视频需区分相机/背景和对象运动，可参考 [Optical Flow in Mostly Rigid Scenes](https://openaccess.thecvf.com/content_cvpr_2017/html/Wulff_Optical_Flow_in_CVPR_2017_paper.html)。这些文献支持辅助模块的可行性，本项目的监督图流程与验收规则为本次设计，尚无实施性能或准确率结果。
