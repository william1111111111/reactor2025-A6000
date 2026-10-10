> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 学生对照恢复执行协议

2026-09-28。用户授权继续中断的四组实验，不改变STUDENT_METHOD_PROTOCOL.md方法或预算。
原源码、PREPARED、所有日志/检查点只读保留；恢复产物在runs/student_method_recovery_v1。

direct_fixed/direct_mode的旧latest.pth均为1660实际优化步，即完整两epoch。
重建前两epoch索引顺序，核对source/view摘要及DataLoader generator状态、scheduler epoch、
optimizer step；恢复模型/优化器/调度器以及CPU/CUDA/NumPy/Python随机状态。
在原GPU1/5重放1661、1662步，与历史loss、各子项和gradient norm逐项精确对照。
如果不一致，停止，不宣称无缝恢复。通过后正式运行仍从1660开始，不使用两步探针权重。
未保存的旧1660步之后更新不混入新正式轨迹，旧日志保留；新history明确继承旧前1660步。

其余kd_basic/kd_residual从原共同初始化开始；四组最终各8300优化步。
共同数据、模型、损失、种子、批量和cosine日程不改。总有效步33200；本次新增29880步，
不包括恢复检查的4步及已中断但未保存的重复计算。实际新增耗时和祖先训练成本分开。

正式队列由systemd用户服务管理，独立于对话终端，失败不自动重试，不重复创建实验。
服务带明确最大运行时限、完整stdout/stderr日志，队列每5秒写PID/阶段/子进程心跳。
状态查询必须交叉检查systemd ActiveState和心跳新鲜度，不能只看静态stage字段。
正式训练、checkpoint核验、VAL571全参考推理/精确DTW和预定四组统计连续执行。
本次只恢复这项已获授权实验，不扩展为新架构、剪枝、多seed或独立TEST。
