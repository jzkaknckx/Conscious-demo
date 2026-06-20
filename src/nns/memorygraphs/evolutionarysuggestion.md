# 激活函数

不同模态相应峰值位置错开
根据模态需要，在CNN的之后一层处加sigmoid或开关函数

# GmemII结构 review时大量特征输入

转化为线性并检索/转化为卷积 ?

# 边缘循迹问题

从特征连续性 -》 interest_map 

#

GI增长问题
子模态分离后 base_interest/conv1*1是否正确

**算法探讨**

当前的代码已经基本上满足让扫视点分块扫视的要求。

0.我做了些许调整，最主要的是将self.semantic_history += semantic_mask_bin其中的semantic_mask_norm替换为了semantic_mask_bin

1.存在看背景的问题：背景中有大面积连续色块，模型很容易选择到背景，需要做出限制。
猜测的可行方案：当V_next = pool(V) * P_surf_map与V = edge_pool(V)扩散完成后，检查最后几圈内是否包含连续的边缘：（1）若被连续边缘包裹则视为正常区域学习（2）若不包含连续边缘或最外侧触及到了人为设置的黑边则视为背景，将此次生成的M_semantic计入semantic_history，但不学习，下个状态仍为MACRO并继续寻找新M_semantic。

2.

3.模型每次扫视时写入的特征中，边缘特征居多。这些特征在Gmem中的存储方式是边缘数值和相对anchor的出现位置，但实际上人类观察过程是看一个连续边缘，比如一个正方体的一条棱
猜测的可行算法调整：