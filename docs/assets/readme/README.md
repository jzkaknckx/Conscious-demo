# README figure provenance / 图片来源

Prepared on 2026-09-23. Figures are copied or decoded from existing outputs without changing their pixels. No results were generated or retouched for presentation.

| File | Source | Scope |
| --- | --- | --- |
| `architecture.png` | Author-supplied `微信图片_20260923145927_186_9.png` | Conceptual architecture, including the earlier saccade controller; not a claim that every illustrated stage is used in the current supervised run. |
| `object_views.png` | `src/test_supervised_graph_learning.ipynb`, code cell at JSON index 5, first PNG output | VOC2007 cat observations 007530 and 000209, normalized views and writable masks. |
| `feature_regions.png` | Same Notebook, JSON cell index 13, first PNG output | Recorded five-modality segmentation for the second demo observation. |
| `familiarity_translation.png` | Same Notebook, JSON cell index 25, second PNG output | Dog 003671, translation condition, from run `voc2007_20260922_222716_908742`. |

The Notebook and actual figure titles determine the observation order; do not infer it from demonstration filenames alone. Original photos are VOC2007 dataset examples, not newly created project imagery. The exported figures are stored here so GitHub rendering does not depend on local absolute paths or Notebook output rendering.

原理图包含早期眼跳设计，README 图注已区分当前实现。其余图片直接提取自实际 Notebook 输出，未修改像素，也未重绘或美化实验成绩。数据照片来自 VOC2007；分区图不是语义分割真值，熟悉度图仅覆盖稀疏模板支持。
