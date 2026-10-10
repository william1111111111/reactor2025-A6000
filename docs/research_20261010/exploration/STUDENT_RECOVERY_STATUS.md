> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 学生对照恢复：独立服务交接

2026-09-28完成更新：四组各8300步，train/VERIFIED、完整571条推理、全参考精确FRD、
eval/analysis.json均已完成，队列stage=complete；独立服务正常成功退出。
下方active/2500步描述是启动时历史记录，不是当前仍在运行。不要再次启动。
结果见STUDENT_METHOD_RESULTS.md，保存入口见ABLATION_INDEX.md。

2026-09-28。已从原direct_fixed/direct_mode的1660步检查点恢复。
旧history最后的1938/1676步不是可恢复模型；旧产物未删除或覆盖。

## 已完成的恢复核验

- 重建前1660步sample/crop/view顺序，两种摘要与旧检查点完全一致。
- DataLoader RNG、optimizer实际1660步、scheduler完成两epoch状态一致。
- 两组各重放1661/1662步；loss、MSE、CCC、velocity与梯度范数和历史日志全部精确一致。
- 恢复测试5项加原24项测试，29项全部通过。
- 正式恢复从旧1660步开始，不使用重放测试权重。

## 持久服务

服务名：`react-student-method-recovery-20260928.service`。
已核验ActiveState=active、SubState=running；服务有独立systemd control group，
不再依赖对话交互句柄。主PID启动时为201355，不应永久假定该PID有效。
最大运行时限4小时，失败不自动重试。仅GPU1/5；GPU6及其他任务未改动。

检查命令（只读，在能访问用户服务管理器的主机环境运行）：

```bash
$LOCAL_PATH/python student_recovery_status.py
```

脚本交叉检查systemd状态、30秒内的心跳及实际文件；进程已停时不会把旧training
字段误报为正在训练。队列每5秒更新`runs/student_method_recovery_v1/status.json`。
进度每100步更新`train/<arm>/PROGRESS.json`；完整epoch结束原子保存latest.pth。
完整训练终点为各组COMPLETE.json，最终评测终点为eval/analysis.json与队列complete。

顺序：恢复两组direct到8300步 → 两组KD从共同初始化训练8300步 → 实际模型审计 →
571条all-session推理 → 全候选精确DTW → 预定四个对比的session聚类统计。
服务输出为`service.log`，各阶段日志位于同一run目录。

本次写记录时，direct_fixed/direct_mode均已超过2500步，并已保存第三epoch检查点；
这不是最终模型或论文效果结论。完成后仍需解读负结果与取舍，不能把自动流水线
结束等同于CVPR投稿完成。原第一轮协议、代码和权重保持只读。
