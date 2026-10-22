# REACT2025：完整序列、双方向、全部 session GT 对照

日期：2026-10-18（Asia/Shanghai）。两个既有冻结模型，未训练、未修改权重。1142/1142 个方向样本已完成，六个分片正常退出。

## 结论

在这套统一 VAL 协议下，集合匹配模型比蒸馏学生的 FRC 高 4.744%，exact FRD 低 5.159%；蒸馏学生的 FRDiv 和 FRVar 更高。此前两组不同协议的历史分数不能支持蒸馏学生优于集合匹配的结论。

## 固定协议

- 同 session 的另一角色全部 GT，paired 放在第一位，未随机抽取 9 条、未删减候选。
- 正向 speaker→listener 571 条；反向 listener→speaker 571 条；双方向 1142 条，按样本等权平均。
- 245–10464 帧完整输入。两个模型均按非重叠 750 帧离线窗口推理，保持 10 个输出 ID，拼接并裁去尾部 padding 后对整段序列评分。不是逐块指标平均，也不是一次无限上下文推理。
- 每个上下文先生成一份官方 Processor 的完整长度 GT（seed 1234），两个模型共用同一个张量。所有预测均进行官方 AU rounding。
- FRC 为每个预测对 GT 取最大 CCC，再对十个预测求和；FRD 对 AU/VA/expression 分组计算无约束精确 DTW，按官方权重组合，对 GT 取最小值并对十个预测求和。
- 推理使用 FP32、关闭 TF32；DTW 为 FP32 输入、FP64 累积，无降采样、无带宽限制。

## 完整结果

| 模型 | 双方向样本数 | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ |
|---|---:|---:|---:|---:|---:|
| 蒸馏学生 | 1142 | 1.039213290 | 142.113959307 | 0.180443333 | 0.067426013 |
| 集合匹配 | 1142 | 1.088513340 | 134.782983228 | 0.152122797 | 0.060046627 |

这里只发布完整序列、双方向 VAL1142 聚合结果。方向子集和中心窗口结果不作为
性能表发布，也不用于方法选择。

## 验证与数值边界

- 1142 条的 audio/emotion/3DMM 特征长度相同，且全部与原视频帧数一致。
- 源文件、GT 成员与顺序、权重、关键实现均在 manifest 中冻结哈希；两个项目的 FaceVerse 归一化文件哈希一致。
- 集合匹配 anchor 在两个运行时中的线性前向路径，用相同冻结权重及变长 mask 输入验证为逐元素相同。
- 最终 10 项分块/精确 DTW 回归测试通过（tests.xml）。
- 真实 3052 帧、13 个 GT、两模型共 20 个预测的完整 FRD，与既有 CPU rolling-DTW 实现逐项相同。
- 114 条短序列控制：连续通道最大预测差异 2.38418579e-06；AU 有 1 个取整翻转（原概率 0.5000000596，接近阈值）。不声称重新推理逐位一致。
- 阈值复查：同一模型/输入，unpadded 概率 0.500000059605，padded 概率 0.499999463558；差 -5.96046447754e-07，解释该 AU 翻转。

## 范围与复现

这是本地 VAL 上两个冻结 checkpoint 的实验；GT 全集是本次明确保留的协议。它不等于官方隐藏 TEST 成绩。非配对 session GT 没有逐帧语义同步标注，仍使用官方长度转换机制。

运行入口：`coreset_reactor.eval_full_sequence_all_session`。`--prepare --output PATH` 冻结清单；`--shard N --shards 6 --output PATH` 分片执行；全部结束后 `--aggregate --output PATH`。GPU 用 `CUDA_VISIBLE_DEVICES` 分配。旧的中心窗口入口及历史分数不覆盖。

本地完整证据：`coreset_reactor/reports/full_sequence_bidir_allgt_20261018/`，包括 comparison.json、deltas.json、逐样本预测/GT 数组、指标、清单、源码快照、各项核验和测试记录。

- student checkpoint SHA-256: `5fc1898b1594da7fe2b4642ad69e051187181de8a8df1432c7a31d4638e7ef1a`
- set_matching checkpoint SHA-256: `ff7c642a3e5aa9ab2effb27fc009498464d040321cbce41a3eef8726837b8553`
