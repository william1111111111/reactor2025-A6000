# 资产核对

## 已确认可复用

- A1 seed 1 对应现有多时间视图学生：`coreset_reactor/reports/multiview_boundary_student_seed1_20261018/student_compute_matched_4epochs/student.pth`。
- 其 config 明确记录 `velocity_weight=0.05`、`relation_weight=0`、4 epoch、8,820 updates；因此它是本轮定义的 Trajectory KD，而不是 A0 Full。
- 多视图教师缓存包含 8,820 个 context、3,320 个源录像、10 个固定教师；缓存 manifest hash 为 `dc2dd0570a9cb5268d854bda97f729f002ca4be6333a48c2a33dceb946fe1f5b`。
- A1 初始化 checkpoint SHA256 为 `ffdc6dabb58a96c521eeacc93c74a81298a93b1e14d1cb7060e06140b235bea4`，初始完整学生 state SHA256 为 `65ee1a21436be9326370bf035836dee49c177325397b996769749133f24f2dd7`。
- A1 的训练样本顺序 SHA256 为 `0d9a679f2a80677c1ed1d7a0e26da7511917a9bb5550db611514bf3fc476fbef`。
- A1 已有完整双方向 VAL1142 原始 p=4 与固定后处理结果，可直接复用；旧的单方向表格不进入本轮结果。

## 本轮已完成

- 匹配的 A0 seed 1：同一缓存、初始化、seed、velocity、4 epoch 和推理流程，只增加中心化残差损失；完成 8,820 updates。
- A0/A1 checkpoint 匹配审计通过：教师列表、缓存、初始化 checkpoint、初始化完整 state、实际样本顺序和训练源码快照一致。
- A0 完整双方向 VAL1142 原始及固定后处理评估；两个方向各 571 条，89/1,142 条触发既定后处理。
- A0/A1 完整 8,820-context TRAIN 缓存差异迁移诊断与 6 个固定哈希示例。
- session-cluster bootstrap 固定 4,000 次、seed `20261023`，两个方向按同 session 共同重采样。
- A3 seed 1 已完成 8,820 updates。它与 A0 的教师缓存、10 个教师、完整初始 state、实际样本顺序、velocity、优化器和更新预算一致；AU/VA/EX 尺度与辅助系数也直接复用 A0。唯一公式变化为不沿 10 个 slot 中心化。
- A2 多时间视图真实参考缓存已完成。逐行审计确认 8,820 个输入 context、view 起点、有效长度和 10 个有序目标身份保持不变；300 个目标张量与生产 loader 逐元素一致。缓存 hash 为 `5f5c338fe27df4502788b76f71a4bf6461453c5384389d4416d6a458f96cfbbd`。
- A2 `velocity=0.05` 与补充的 A2/A1 `velocity=0` 两组均完成 8,820 updates；对应 A1 对照的初始 state、输入顺序、架构、超参数和更新数审计通过。
- A0/A1/A3 的 8,820-context 软 AFR 诊断完成；A0 相对 A3 在 AU、VA 的中心化残差 MSE 分别降低 6.52%、4.09%，EX 高 0.10%，三组 pair-distance MAE 均更低。A0/A3 初始加权辅助梯度相对基础梯度分别为 0.252 和 2.206，因此 A3 的相同系数不构成相同梯度强度。
- Fixed、Uniform、Target-specific 三个 reference 教师库的本地资产审计通过。三组共享 query 0；rank 1–9 的固定目标 manifest、完整初始 state、600-update 样本顺序、模型配置和优化配置匹配，三组非 query-0 checkpoint 确实不同。Target-specific 十个 checkpoint 哈希与当前 8,820-context 多时间视图缓存逐项一致。教师库组合哈希分别为 `9b37c9b8631dfc6451d603ff7565c80b9c29d10a5f24d8b23086bdcb6bfc4c60`、`0f955a6679375bdfea9565e54a2f7b272d327ff3c896b71bffa11a05270e8692`、`b9539b476651f839ed73acedee7d5b01c6821a305edb59b86de4127e07d39b38`。
- A2 velocity、A2/A1 no-velocity 与 A3 均已完成完整双方向 VAL1142 原始/既定后处理评估；全部训练匹配、A2 缓存身份与正式评估人口审计通过。机器可读汇总为 `SEED1_CONTROLS.csv/json`。
- A0/A1 的预定 seed 2、3 均完成 8,820 updates 与完整双方向 VAL1142 原始/既定后处理评估。每个 seed 内的教师库、缓存、初始化 checkpoint、完整初始 student state、实际输入顺序、源码快照、velocity 和更新数审计全部通过；三种子汇总为 `MULTISEED_A0_A1.csv/json`，人工可读表为 `MULTISEED_REPORT.md`。
- Fixed 与 Uniform 的 8,820-context 多时间视图教师缓存已构建；它们与 Target-specific 缓存的有序 source views 哈希同为 `cc88d3d3ee0c4c3c7d5df46cbd7fee2640bb1219d0427e49bfd9788ffa5bdc30`，缓存/教师库审计通过。
- Fixed 与 Uniform 对应学生已完成 8,820 updates，Target-specific 复用 A0 seed 1。三组完整初始 state、实际输入顺序、架构、数值尺度、关系项系数、velocity 与更新数审计通过。
- Fixed、Uniform、Target-specific 三套教师库已在同一次完整双方向 VAL1142 流程中完成原始及既定后处理评估；28 个唯一 checkpoint 只各执行一次。既定后处理的 `(FRC, exact FRD, FRDiv)` 分别为 Fixed `(0.797126, 135.220627, 0.202366)`、Uniform `(1.537155, 134.646075, 0.101901)`、Target-specific `(1.050503, 133.195375, 0.190064)`。
- Fixed、Uniform、Target-specific 三个对应学生均已完成完整双方向 VAL1142 原始/既定后处理评估与 4,000 次 session-cluster bootstrap。既定后处理的 `(FRC, exact FRD, FRDiv)` 分别为 Fixed `(0.837835, 134.485993, 0.190907)`、Uniform `(1.574821, 137.720230, 0.091454)`、Target-specific `(1.101366, 134.225506, 0.178959)`；联合机器汇总为 `REFERENCE_CONTROLS.csv/json`，人工可读报告为 `REFERENCE_CONTROLS_REPORT.md`。

## 源码对应结论

- 当前 `relation_distillation.py` 确实实现 `centered_residual_loss`，不是以余弦方向损失代替。
- 当前训练入口能分发 `centered_residual`，并记录固定 TRAIN 尺度、权重、源码哈希、缓存哈希、初始化哈希和样本顺序哈希。
- `test_relation_distillation.py` 已覆盖：教师等于学生时为零、共同轨迹平移不变、沿 slot 中心化、padding 排除、残差幅度敏感、教师无梯度和 slot 对应关系。连同 frozen calibration、velocity mask 和 overlap/postprocess 测试，本轮最终共 23 项通过。

## 排除项

- 旧单视图 A0/A1 只用于确认损失配方和历史校准，不能作为本轮正式对照。
- 当前发布的多视图主结果不能标成含中心化差异项；其 provenance 显示关系项为零。
- REACT2024 的任何缓存、训练或评估均不进入本目录。
