> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 第二轮：匹配监督对照（预注册）

2026-09-27。以 PROTOCOL_AUTHORITY_CORRECTION.md 为最高项目评测说明。
REACT2025 主评测使用论文定义的 all-session 参考集合，不改为抽样十参考。

## 问题与四组

仅改变 CCC 项的参考权重，固定 MSE 与速度项的目标不变。

1. fixed：固定目标 one-hot 权重，单参考 CCC。
2. uniform：完整 TRAIN 参考集合的均匀权重。
3. mode：TRAIN 参考的均值/标准差描述子距离生成 Gibbs 权重，有效支持数8。
4. shuffled：对 mode 权重的参考列作固定随机循环置换；保留每行权重值及熵，
   破坏权重与实际参考行为模式的匹配。置换仅由 TRAIN group 与种子27091决定。

所有组都计算完整参考 CCC 矩阵、同一 mode prior，再施加各自权重。
汇聚函数均为 tau*logsumexp(CCC/tau+log_prior)，tau=0.05。
使用已修正的 CCC 核：常数参考贡献0，但常数预测对变动参考仍有有效梯度。
训练参考相对时间重采样不等于评测 Processor；两者明确区分。
这是已有监督设计的机制对照，不能预先宣称新算法或CVPR录用级贡献。

## 固定项、预算与验收

- 同一归档 REGNN 源码、参数3720129、seed1、相同初始化。
- 每组9个 specialists，rank1..9；固定目标采用旧 T1 manifests，冻结 paired q0
  共享于四组。推理不接触 GT，参考库只用于 TRAIN loss。
- 每个 specialist 600 updates；batch32，clip750，BF16，AdamW1e-4，50epoch
  cosine 日程，CCC权重1，velocity权重0.05；不择优挑步、不用 VAL 调参。
- 共21600新优化步。卡1和5各最多一个作业，卡6不占用。每作业最长1800秒；
  失败保留产物并停止调度新任务，不能默默缩减样本、参考或训练步数。
- 先四组 rank1 各2步 smoke，检查损失/梯度有限、初始化及 source schedule
  digest完全一致。正式36作业每个都核验初始化及600步数据日程哈希。
- 所有组的全参考比较计算相同，但不同先验操作和缓存会带来耗时差异；
  保存实测时间/显存，不声称精确相同FLOPs或墙钟。
- 输出仅位于 $EXPLORATION_ROOT/runs，旧源码、模型、数据只读。

## 评测与决策

- 完整 VAL571，统一 paired source 顺序、all-session GT cache、25D AU取整及变长处理。
- 报告 FRC、精确 FRD、FRDiv、FRVar；同步指标及模式覆盖指标需先核验实现，
  不用配对CCC替代集合恰当性。冻结 q0 应在四组间逐元素相等。
- 按session聚类的配对bootstrap，报告四组全部结果与每候选结果。
- uniform vs fixed 检验集合监督；mode vs uniform 检验模式保持；mode vs shuffled
  检验模式匹配是否必要。不能单看某个FRC提升就认定成功，需联合质量/多样性。
- 单训练种子的探索性VAL证据，不是独立TEST或正式SOTA。若机制对照不支持，
  记录负结果并调整研究动机，不通过补调超参掩盖结果。
