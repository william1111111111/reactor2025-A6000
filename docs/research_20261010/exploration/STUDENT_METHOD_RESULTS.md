> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 完整学生对照：残差改善 FRD，但蒸馏优势仍是质量—多样性取舍

2026-09-28。四组训练、模型审计、VAL571/all-session 推理、全参考精确 FRD 与
session bootstrap 已完成。最终产物在 `runs/student_method_recovery_v1/`。
队列 `status.json` 为 complete；原中断目录仍完整保留，不重复启动。

## 固定条件

四组相同 3,803,201 参数、十个输出、共同配对预训练骨干初始化、3320 TRAIN views、
样本顺序和 8300 优化步。不是纯随机从零训练，不是总训练成本相同：KD 另需教师训练
与缓存。两组 direct 从旧 1660 步恢复并通过精确重放，KD 两组从共同初始化开始。
原协议、源文件哈希、恢复祖先记录和实际优化器步数均保留。

## 结果

| 方法 | FRC ↑ | 精确 FRD ↓ | FRDiv ↑ | FRVar | soft FRDiv |
|---|---:|---:|---:|---:|---:|
| direct_fixed | 1.074794 | 97.978659 | 0.179503 | 0.064712 | 0.081029 |
| direct_mode | 1.813641 | 95.659014 | 0.096420 | 0.069750 | 0.038855 |
| kd_basic | 1.464839 | 100.774094 | 0.171190 | 0.070300 | 0.086928 |
| kd_residual | 1.447309 | 97.606049 | 0.173210 | 0.068254 | 0.086741 |

全部四项预定比较，不只保留有利结果：

| 处理−控制 | ΔFRC [95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |
|---|---|---|---|
| direct_mode−direct_fixed | +0.738847 [0.495779, 0.993585] | −2.319645 [−3.780237, −0.239169] | −0.083083 [−0.101620, −0.064084] |
| kd_basic−direct_mode | −0.348802 [−0.592882, −0.132087] | +5.115081 [3.215196, 6.268887] | +0.074770 [0.059058, 0.092800] |
| kd_residual−kd_basic | −0.017530 [−0.031107, −0.000427] | −3.168046 [−3.689838, −2.105217] | +0.002019 [0.000605, 0.003334] |
| kd_residual−direct_mode | −0.366332 [−0.623718, −0.139798] | +1.947035 [0.616729, 3.133586] | +0.076790 [0.059797, 0.095930] |

20 个 session 聚类、571 contexts 加权、5000 bootstrap、seed270927；
区间不包含训练种子不确定性，未做多重检验校正。

## 解读与下一步边界

- residual 相比 basic 确实降低 FRD，但 FRC 轻微下降；不是全面优势。
- residual 的离散 FRDiv 小幅上升，但 soft FRDiv 差异 −0.000186，
  区间 [−0.001636, 0.001640] 跨零；不能主张连续反应多样性已经改善。
- residual 相比 direct_mode 有更大输出间多样性，但 FRC/FRD 更差。
  高 FRDiv 本身不证明有效模式覆盖，不证明更好的 Pareto 边界。
- 全部是单 seed 的探索性 VAL；没有独立 TEST、新多 seed 或新外部基线。
- all-session 参考集合符合 REACT2025 论文定义；本地 Processor 后 25D 评分与论文
  latent-space 措辞之间的实现边界仍在，不声称逐细节官方复现或论文 TEST 排名。

完整机器结果：[analysis.json](runs/student_method_recovery_v1/eval/analysis.json)。
权重审核：`VERIFIED.json` (local reference: `runs/student_method_recovery_v1/train/VERIFIED.json`)。
保留索引：[ABLATION_INDEX.md](ABLATION_INDEX.md)。
