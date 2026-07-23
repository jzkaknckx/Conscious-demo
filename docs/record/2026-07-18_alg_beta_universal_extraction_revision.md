# 2026-07-18 Alg_beta 多模态抽取与相似度规范修订

- 在 `docs/algorithm/Alg_beta.md` 结尾续写“探讨二修订版”，审查原 Universal Extraction Pipeline 中通道分类和检索方法的缺陷。
- 将 `grad` 模态规范为 `[abs_strength, theta, gradx, grady]` 四通道，其中强度与角度只参与门控/上下文，`gradx` 与 `grady` 参与相似度检索。
- 补充 `grad`、`RGB`、`Curv`、`aps`、`ori` 五类输入的子通道依赖、父门控和拓扑职责。
- 补充分组相似度检索公式、分通道参数要求、schema 配置建议和最小落地顺序。

