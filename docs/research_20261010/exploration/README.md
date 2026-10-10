> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT → CVPR2027 exploration

优先阅读：[论文定义与评测结论更正](PROTOCOL_AUTHORITY_CORRECTION.md)。
REACT2025的all-session参考集合有论文直接依据；不再按代码抽样替换主评测，
也不把同session置乱不降分当作反应集合无效的证据。以下为历史第一轮记录。

原项目只读：`$LOCAL_PATH/react2025_new`（实际 `$PROJECT_ROOT`）。

最新入口：[结构与组合消融执行状态](COMPONENT_ABLATIONS_STATUS.md)、[冻结协议](COMPONENT_ABLATIONS_PROTOCOL.md)、[消融保留总索引](ABLATION_INDEX.md)。
2026-09-30：用户授权两项都做，暂不增加随机种子。新增集合匹配加动态约束、十输出头结构对照，
独立保存到 `runs/component_ablations_v1`。57项测试、3320输入核验、三组旧对照精确重放、
两组GPU及端到端smoke通过，独立服务已启动GPU5/1正式训练，各8300步后自动评测。
本段是启动快照，实时状态用 `component_ablations_status.py` 查询，禁止重复启动。

上一轮：[三种新蒸馏完整结果](runs/distillation_methods_v1/eval/RESULTS.md)、[执行记录](DISTILLATION_METHODS_STATUS.md)。
三组均已8300步完成，VAL571/all-session评分与统计完成，服务成功退出。
assignment改善FRC和多样性；dynamic降低FRD但损失多样性，不宣称全指标优越。
旧dynamic触及预设系数上限，初始梯度比仅2.7%–4.8%；这个数不能当作新组合组的测量值。
所有历史结果保留，不重复启动旧服务。

前一轮：[不同蒸馏方式调研](DISTILLATION_REVIEW_20260928.md)、[完整学生结果](STUDENT_METHOD_RESULTS.md)。
2026-09-28：四组学生均已8300步完成，VAL571/all-session精确评分和统计已完成。
全部历史消融（含负结果和中断记录）原位保留；当时的调研方案现已另目录实施。

执行记录：[恢复服务状态](STUDENT_RECOVERY_STATUS.md)、[原执行记录](STUDENT_METHOD_STATUS.md)、[冻结学生对照协议](STUDENT_METHOD_PROTOCOL.md)。
原句柄25075已中断，旧full_status.json已失真，不作为存活依据。2026-09-28恢复到
`runs/student_method_recovery_v1`，由`react-student-method-recovery-20260928.service`
独立托管；使用`student_recovery_status.py`检查服务与心跳，禁止重复启动。
两组direct从1660步恢复并通过历史1661/1662步精确重放，另两组KD从共同初始化训练；四组均已完成。
3320输入核验、350目标一致性检查、24项测试和端到端GPU smoke均已通过。
结果见STUDENT_METHOD_RESULTS.md；不把本轮初始化称为纯随机从零训练，不把9.78倍参数压缩等同加速。

第二轮实验和统计已完成：[研究结论](SECOND_ROUND_REPORT.md)、[执行记录](SECOND_ROUND_NOTES.md)、[匹配监督协议](MATCHED_SUPERVISION_PROTOCOL.md)。
四组对照保存到 `runs/matched_supervision_v1`：36个600步训练已完成，
`train/VERIFIED.json`及`train/CHECKPOINT_AUDIT.json`核验通过；训练句柄19047已正常退出。
完整571条推理也已完成，`eval/INFERENCE_AUDIT.json`确认四组共享q0与历史预测逐元素一致。
四组571条精确FRD评分已完成，句柄78570正常退出；matched_analysis.py也已完成。
uniform提高FRC但降低输出间多样性；mode保留多样性却未胜过打乱对应控制，
没有证明更好的质量—多样性边界。不直接扩大当前mode方案训练。
2024原生neighbor资产行映射仍未认证，不宣称已得到2024 native分数。
训练、分析及首轮审计合计18项单元测试通过。
[监督机制解释](SUPERVISION_MECHANISM.md)、[最新文献边界](LITERATURE_UPDATE_20260927.md)。

以下为第一轮归档状态：

2026-09-27首轮完成：[研究报告](REPORT.md)。
核心发现：冻结模型保留可测的配对条件信息。同session共享参考集合时，交换输入
对应关系可能不改变集合分数；这是论文任务定义下可能出现的预期性质，不能据此
断言生成反应不恰当、官方指标失效或新方法达到SOTA。

- 结果和限制：`REPORT.md`
- 预定实验与统计澄清：`PROTOCOL.md`
- 最终统计：`runs/final_analysis_v1/summary.json`
- 单元测试：7 passed
- GPU作业：已完成，无本轮后台训练

继续工作以 PROTOCOL_AUTHORITY_CORRECTION.md 为准：保留论文定义的all-session
主评测，开展匹配监督控制实验，检验质量、多样性与模式覆盖。旧目标中的
“REACT2025固定十参考评测协议”措辞已被用户纠正，不再作为主评测要求。
