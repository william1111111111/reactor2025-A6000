# REACT2024 alpha=1.50 官方五指标结果

完整 VAL1124，冻结 count4 学生，同 session 另一角色的全部 GT。预测先执行
slot-centered spread `alpha=1.5`，再执行官方范围投影与 AU rounding，最后对
query 0 的每个常量 AU 加一个与源 AU 变化位置对齐的转换帧。评估函数保持
Mam-Reactor 官方定义不变。

| 方法 | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | FRSyn/TLCC ↓ |
|---|---:|---:|---:|---:|---:|
| 原始学生 + 稀疏 AU 转换 | 0.655362986 | **86.025618895** | 0.132453367 | 0.058834873 | **44.733096085** |
| `alpha=1.5` + 稀疏 AU 转换 | **0.703215866** | 93.945096669 | **0.168626100** | **0.071234852** | 44.849644128 |

`alpha=1.5` 提高 FRC、FRDiv 和 FRVar，同时使 exact FRD 增加 7.919477773，
FRSyn 增加 0.116548043。FRSyn 越低越好，因此缩放没有改善同步指标。

## FRSyn 与后处理审计

完全不做稀疏转换时，原始与 `alpha=1.5` 的官方 TLCC 都是 49。原因是官方
实现只返回 query 0，并且任一常量通道都可能使相关系数成为 NaN，随后把
最优 lag 固定到搜索边界。VAL1124 中有 983 条源表情本身含常量通道；稀疏
转换实际改变 141 条预测。后处理后仍有大量上下文位于边界值，因此 FRSyn
仅按官方协议报告，不把它解释为稳健的十输出同步性。

稀疏转换只改预测，不修改 GT 或评估函数。常量 0/1 分别在对应源 AU 的内部
最大/最小位置翻转一帧；若源表情任一通道恒定，则保持预测不变。该操作使
FRC、exact FRD、FRDiv 和 FRVar 相对转换前只发生约 `1e-4` 或更小的变化。

## 缩放归因

转换前的校准归因实验得到：baseline FRC 为 0.655348190；缩放预测固定使用
baseline 的 GT 匹配时为 0.692372901，重新进行 all-session GT 匹配时为
0.703201093；18.950% 的候选改变了最佳 GT。约 77.4% 的 FRC 增益在固定旧
GT 匹配时已经存在。该归因只分析缩放本身，不包含之后增加的稀疏转换。

这是验证集校准结果，不是独立 TEST 成绩。FRDvs 与 FRRea 未评估；本地
Mam-Reactor 的 FRDvs 入口是占位实现，不能提供有效分数。机器可读结果位于
`published_results/react2024_accepted_operating_point.json` 和
`published_results/current_official5_results.json`。
