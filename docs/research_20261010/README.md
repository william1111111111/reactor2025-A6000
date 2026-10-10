# REACT 研究记录与完整实验汇总

发布整理日期：**2026-10-10**。本次按用户要求上传本地已有的实验结果、研究过程、参数与复现记录；没有重新训练或重跑全量评估。

## 阅读入口

1. [当前方法与完整关键参数](historical/archive_20260930/METHOD_AND_PARAMETERS.md)：固定多样 GT、模式保持多参考教师、共享多输出学生、损失、优化器、预算与变长处理。
2. [主结果、历史路线、负结果与勘误](historical/archive_20260930/RESULTS_AND_HISTORY.md)：REACT2025/2024、M1/M2/M3、teacher/学生、graph 对照、随机参考与输入依赖诊断。
3. [复现定位与权重 hash](historical/archive_20260930/REPRODUCIBILITY.md)、[冻结 operating points 的机器记录](archive_20260930/OPERATING_POINTS.json)。路径变量需在本地配置；权重本身不上传。
4. [学生 direct/KD matched 对照](exploration/STUDENT_METHOD_RESULTS.md)、[不同蒸馏方法](exploration/runs/distillation_methods_v1/eval/RESULTS.md)、[结构与组合消融](exploration/runs/component_ablations_v1/eval/RESULTS.md)。这些是**独立探索批次**，不要与主归档混作相同初始化的一次 sweep。
5. [完整记录目录](INDEX.md)：202 个来源记录、189 个去重后的报告/聚合 JSON；[来源及发布文件 hash](MANIFEST.json)。包括历史 pilot、失败、停止、恢复和未完成计划，而不是只筛选正结果。

## 两组结果要分开读

9 月 30 日归档中建议保留的 **Base KD + centered-residual KD** operating points：

| 数据集/口径 | contexts | FRC ↑ | exact FRD ↓ | FRDiv | FRVar |
|---|---:|---:|---:|---:|---:|
| REACT2025 all-session VAL | 571 | 1.445507 | 97.463281 | 0.170130 | 0.067918 |
| 转换版 REACT2024 all-session VAL | 1124 | 0.656394 | 86.033383 | 0.132724 | 0.058911 |

独立 exploration 批次的完整 **REACT2025 VAL571/all-session** 结果：

| 方法 | FRC ↑ | exact FRD ↓ | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| Basic KD | 1.464839 | 100.774094 | 0.171190 | 0.070300 |
| Whole-clip assignment | 1.483930 | 100.578430 | 0.174516 | 0.072014 |
| Dynamic residual | 1.453440 | 95.718811 | 0.167214 | 0.066872 |
| Assignment + dynamic | 1.471408 | 96.182289 | 0.166247 | 0.067881 |
| Shared backbone + ten linear heads | 1.396709 | 96.999068 | 0.188895 | 0.068924 |

原始聚合数值、全部预定 contrasts、session-bootstrap 区间和匹配训练审计见
[analysis.json](exploration/runs/component_ablations_v1/eval/analysis.json)，
参数量和推理微基准见 [BENCHMARK.json](exploration/runs/component_ablations_v1/eval/BENCHMARK.json)。
三组恢复/后续实验的最终状态均为 complete；历史 STATUS 文档中的“启动/尚未完成”是当时快照，不是当前状态。

这组结果支持质量与多样性的取舍，不支持“每个模块缺一不可”或所有指标都改善。
微基准不是训练、数据读取和评估在内的端到端加速；教师训练成本也不能省略。

## 评估口径与结论边界

- all-session GT 集合依据与第一轮表述更正见 [PROTOCOL_AUTHORITY_CORRECTION.md](exploration/PROTOCOL_AUTHORITY_CORRECTION.md)。十个生成输出不等于只允许十条参考 GT。
- 主结果使用有效 source prefix、AU round、固定随机种子的官方 learned GT 后处理、逐预测 best-GT FRC / min exact grouped-DTW FRD。两指标可选择不同 GT。
- FRC 对十条预测求和，因此可以大于 1；FRDiv 是预测之间的距离，与参考 GT 数量无关。
- 本地 Processor 后 25D 评分与论文 latent-space 措辞的对应仍须明确区分；不据此声称挑战服务器逐细节复现或官方 TEST 排名。
- REACT2025 当前本地 VAL/TEST 内容相同，只计一套评估。最终主线主要是单训练 seed；早期 M1 三 seed 不能替代最终方法的多 seed 证据。
- 2024 为转换数据下的 all-session 分析，不自动等价于其原生 neighbor 协议。不能把两个数据集或不同参考池的 FRC 直接横比。
- session bootstrap 不代表训练种子方差；历史 oracle、可行下界、数值上界、部署模型分开标注。

## 发布范围与保密边界

本次是**报告与聚合证据发布**，不是全部工作区源码同步，也不是完整离线恢复镜像。当前未提交的训练源码、权重、数据、教师缓存和逐样本记录仍保留本地；源码/权重 hash 帮助定位，不能当成已提供对应文件。复现文档中的代码入口可能尚未全部出现在当前远端源码中。

没有上传数据集、音视频、25D/58D 轨迹、checkpoint、评估后处理权重、缓存、SSH/API 凭据、逐样本映射、私有服务器地址或整个 LOCAL ONLY 归档。此前要求保密的 Mam GT-oracle 启动包/实验不在导出范围。

本机绝对路径替换为 `$PROJECT_ROOT`、`$RUN_ROOT`、`$SOURCE_ROOT`、`$EXPLORATION_ROOT` 等占位符；长数组与样本级字段不发布，字段省略记录见 manifest。原始日志未上传；研究过程通过协议、阶段报告、审计摘要、恢复记录和负结果保留。

历史文档中“本次不 push / LOCAL ONLY”指原始本地归档；**本目录是本次另外生成、经过脱敏审查的公开版**。历史“当前方法/未启动”也以当时日期为准。本目录不自动选新主模型、不删除旧结果、不启动新实验。

[PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) 记录这次文件 hash、表格和隐私扫描检查；它证明导出一致性，不证明所有历史科学解释正确。
