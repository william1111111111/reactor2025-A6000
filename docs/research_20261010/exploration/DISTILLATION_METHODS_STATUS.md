> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 三种新蒸馏：执行与交接

2026-09-30核实：本轮三组训练与完整VAL571/all-session评测已完成，服务成功退出，
最终结果见 [完整结果](runs/distillation_methods_v1/eval/RESULTS.md)。不要重启此服务。
新一轮结构与组合消融见 [最新执行状态](COMPONENT_ABLATIONS_STATUS.md)。

以下保留2026-09-28的启动与交接记录，其中训练步数属于历史快照，不是当前进度。

## 已通过

- 新增12项损失测试，连同历史测试共41项，通过主机测试和服务内再次测试。
- 3320个TRAIN source/crop/view、完整教师cache shard SHA、概率语义、初始模型校验。
- 普通KD和残差KD各重放两步：所有日志标量，以及模型、优化器、scheduler、
  CPU/CUDA/NumPy/Python/DataLoader随机状态与历史两步checkpoint完全一致。
  因此本轮允许复用旧两组完整8300步结果；新三组的完整数据日程仍须在训练结束核验。
- 新三组各两步真实GPU smoke、checkpoint审计、三条VAL的推理与全参考精确评分通过。
  smoke数字只是程序检查，不解释为效果。
- 独立服务已启动正式训练；最近一次人工核对，typed和assignment均已超过100步，
  dynamic排在下一波。这是快照，实时状态以检查脚本为准。

## 执行范围

新目录：`runs/distillation_methods_v1/`。
服务：`react-distillation-methods-20260928.service`，独立systemd control group。
只用GPU1/5，6及其他作业未动。第一波typed/assignment，第二波dynamic，
各8300步，总24900个新正式优化步；随后自动进行VAL571/all-session精确评分和统计。
最长六小时，单作业两小时，失败停止、不自动重试。禁止重复启动。

只读查询（需要可访问主机用户服务总线）：

```bash
$LOCAL_PATH/python distillation_methods_status.py
```

`status.json`每5秒心跳；各train/<arm>/PROGRESS.json每100步更新，
完整epoch原子保存latest.pth及optimizer/scheduler/RNG，COMPLETE.json才表示训练完成。
最终需要队列stage=complete、eval/analysis.json和eval/RESULTS.md都存在。
服务正常完成后inactive不是中断，不能只凭静态stage或历史PID判断运行。

## 冻结校准与解释限制

配置/源码/协议SHA在PREPARED.json；旧协议与模型没有修改，所有结果分目录保存。
typed辅助系数0.05607332367787542；四个初始TRAIN probe实际梯度比例约20%–37%。
dynamic系数达到预设上限1，实际比例仅约2.7%–4.8%，没有达到25%目标。
保留这个上限与实际比例，不在看到VAL之后临时调参；本轮不能声称所有辅助项严格
等梯度强度，也不能用dynamic无改善直接排除较强动态监督。若后续研究权重，另列新消融。
assignment只有整段匹配变化，没有新增辅助系数，q0只固定对应、不冻结参数。

最终合并旧baseline前，检查clip ID、session、GT数量/缓存、评分源码SHA；
报告五个预定对照及连续残差分解，不只挑有利指标。仍是单seed VAL，不是独立TEST。
新训练代码不直接支持盲目自动resume；中断应先审核epoch checkpoint和随机状态，
在新的恢复目录处理，不覆盖失败记录。

沙箱内一次历史多进程测试因本地套接字权限受限卡住，已结束对应三个测试进程；
主机环境和独立服务内测试均通过。该问题不涉及模型训练失败或改动其他用户任务。

入口：[冻结协议](DISTILLATION_METHODS_PROTOCOL.md)、[保存索引](ABLATION_INDEX.md)、
[先前方案调研](DISTILLATION_REVIEW_20260928.md)。
