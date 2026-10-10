> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 现有方法验证：执行状态与交接

2026-09-28最终更新：恢复目录runs/student_method_recovery_v1的四组训练、审核、
VAL571精确评测与统计全部完成，详见STUDENT_METHOD_RESULTS.md。
本文件下方为原执行历史；消融全部保留，下一轮蒸馏方式尚未开始训练。

2026-09-27。用户要求沿现有方法验证投稿价值，明确不转向剪枝。

2026-09-28更新：下述原队列已中断，句柄25075失效，旧full_status.json的training
不是存活证据。旧日志/检查点全部保留。恢复已转入独立目录与systemd用户服务，
详见STUDENT_RECOVERY_STATUS.md；以恢复服务及其心跳为当前状态。

## 本轮范围

第一项关键对照：同大小直接学生 vs 基础KD vs 现有残差KD。
新增direct_fixed/direct_mode两个直接训练对照；四组3,803,201参数、相同配对模型
初始化、相同3320 TRAIN views与8300更新。不是纯随机初始化，也不是总训练算力相同：
KD额外依赖已经训练的教师与其缓存。规则见STUDENT_METHOD_PROTOCOL.md（已冻结）。

## 已验证

- 全部3320缓存source/crop/长度一致性，cache shards与TRAIN参考SHA已记录。
- 35个抽查输入×10槽位=350个raw目标，与原canonical加载器逐元素一致。
- 初始学生state哈希与历史学生初始化一致，历史TRAIN残差校准来源核验通过。
- 新增6项测试、历史18项测试，合计24项通过。
- 四组各2更新GPU smoke完成；梯度、模型、优化器有限，实际step=2。
- 四组初始化、source顺序、crop/view顺序哈希相同。
- 三条VAL样本的四组推理与全参考精确DTW评分通过；smoke数字不是有效方法结论。

## 原正式队列（已中断，以下是历史记录，不重启）

启动命令：

```bash
$LOCAL_PATH/python run_student_method.py --phase full
```

交互句柄25075。两轮训练：GPU1/5先direct_fixed/direct_mode，再kd_basic/kd_residual。
随后自动运行实际checkpoint审核、571条推理、40候选/输入精确DTW、四个预定对比的
session聚类bootstrap。失败会写状态并停止后续阶段，不自动降低样本数或训练预算。

状态：`runs/student_method_v1/full_status.json`。
日志：`full_direct_fixed.log`、`full_direct_mode.log`、`full_kd_basic.log`、`full_kd_residual.log`，
以及后续`full_audit.log`、`full_infer.log`、`full_score.log`、`full_analysis.log`。
全部日志均位于`runs/student_method_v1`。

训练中每个arm的`train/<arm>/history.jsonl`记录每一步；`latest.pth`仅在epoch结束保存，
不要把该文件存在解读为作业完成。`COMPLETE.json`才是该训练任务的终点。
最终流水线完成需`full_status.json`的stage=complete及`eval/analysis.json`。
若失败，先读错误和已有产物；拒绝覆盖保护是有意设计，不能盲目重启。

## 仍需人工/后续分析

流水线跑完后必须解读四组结果、全部四项预定对比与取舍，不能把任务启动当作实验成功。
多seed、独立TEST、强外部基线、严格性能基准仍未完成；未自动开启这些更大实验。
上一轮四组教师监督消融已完成，见SECOND_ROUND_REPORT.md，不重复跑它。
原项目未修改，GPU6未占用，历史checkpoint/结果没有删除或覆盖。
