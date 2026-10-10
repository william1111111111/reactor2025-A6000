> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 四组监督对照到底检验什么

本说明不改变已冻结的训练配置，也不依据部分VAL结果挑选模型。

## 基线不是随机单参考

本轮固定目标来自既有TRAIN-only T1 manifest：保留原配对q0，并在轨迹空间选择
Max-FRDiv的另外9个目标，描述子再分配slot身份。因此fixed是一套已有模式覆盖
设计上的固定单参考损失，不是随便抽一个目标的弱基线。四组共享同一T1映射。
本轮检验CCC参考汇聚的增量作用，不重新验证T1选择本身优于随机T0。

来源：归档 `coreset_reactor/maxfrdiv_teacher_manifest.py` 顶部说明及T1 manifest
的 `assignment_method`、`preserve_native_pair`、`source_manifest_file_sha256`。

## 汇聚的精确含义

设完整参考集合上某个预测的CCC为 s_j，先验权重为 pi_j，温度tau=0.05。
本轮汇聚为

    A(s, pi) = tau log sum_j pi_j exp(s_j/tau)
             = max_q [sum_j q_j s_j - tau KL(q || pi)]

最优权重 q_j 正比于 pi_j exp(s_j/tau)，且 dA/ds_j = q_j。
这是log-sum-exp的变分恒等式，不是本工作的数学创新；已做数值与梯度单元测试。

- fixed：pi是固定目标的one-hot，退化为该目标CCC。
- uniform：pi在完整参考集合均匀，但梯度权重q并不均匀，仍可集中到当前高分参考。
- mode：pi偏向固定目标邻近的行为模式，在软匹配时限制跨模式漂移。
- shuffled：保留pi各行的数值和熵，但打乱参考对应，检验行为几何而非仅先验尖锐度。

上述q只表示损失内的参考责任权重，不是经过校准的真实反应概率。
“全部参考有效”不要求训练对每个参考给出相同梯度，也不意味着最优输出应为参考平均。
均匀汇聚可能产生模式集中是一个待证机制，不是仅凭公式就证明实际模型坍缩。

## 结果解释边界

MSE的固定目标仍提供模式锚点；本轮不是完全set-only训练，也不移除原配对q0。
四组使用相同的自定义Conditional-REGNN骨干，不宣称它等同原论文REGNN实现。
不能直接据此认为该方法适用于所有扩散/flow骨干，或已在独立TEST证明泛化。

模型及优化器36份完整checkpoint内容审计通过；四组571条预测的q0与历史冻结q0
逐元素一致。完整集合指标及session聚类统计现已完成，见SECOND_ROUND_REPORT.md：
mode改变质量—多样性取舍，但未证明优于shuffled，也未证明独立的有效模式覆盖优势。
