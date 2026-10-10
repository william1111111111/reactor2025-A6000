> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 实验结果与研究历程

归档截至 2026 年 9 月 30 日。下面是面向论文写作的精选汇总；**所有扫描到的原始报告、JSON、配置和日志另存 evidence，完整检索见 ALL_REPORTS.md 与 metric_index.csv**。这里不将 pilot、不同口径、旧 checkpoint、oracle 和部署方法拼成一张排行榜。

## 当前教师与学生主结果

REACT2025：完整 VAL571，all-session opposite-role GT，AU round，确定性 learned target processing，exact DTW，seed1。

| 角色 | FRC ↑ | exact FRD ↓ | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| Mode-reference 教师 count8 corrected | 1.370792 | 95.772670 | 0.190244 | 0.069174 |
| Uniform-reference 教师 corrected control | 2.038626 | 95.410771 | 0.091645 | 0.070391 |
| 学生 Full KD | 1.444718 | 97.540968 | 0.170190 | 0.067888 |
| 学生 Minimal KD | 1.454643 | 99.714605 | 0.170338 | 0.069951 |
| 学生 Velocity only | 1.454373 | 99.752944 | 0.170335 | 0.069890 |
| 学生 Residual only 建议简化版 | 1.445507 | 97.463281 | 0.170130 | 0.067918 |
| 相同固定真实 GT 直接监督学生 | 1.066966 | 97.427886 | 0.177390 | 0.064775 |

Teacher prior 对照仅改变先验；Direct GT 对照应与 Minimal KD 比，不与 Residual-only 充当单变量对照。学生对教师保留约 89.4% FRDiv，FRC 较高而 FRD 较差，不能称全面超越教师。

来源：`react2025_key_component_ablations_seed1/summary.json` 与 `react2025_student_loss_pruning_seed1/summary.json`。两者的 mode bank 均为重新评估后的 corrected 数值，不用原 uniform-v1 替换。

转换版 REACT2024：完整 VAL1124，相同 all-session 分析口径，非其原生 challenge protocol。

| 角色 | FRC ↑ | exact FRD ↓ | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| Mode-reference teacher count8 | 0.657408 | 85.626572 | 0.146408 | 0.058906 |
| Mode-reference teacher count4 | 0.648193 | 86.151776 | 0.152807 | 0.060486 |
| count8 对应 Full 学生 | 0.671193 | 85.730786 | 0.127287 | 0.057549 |
| count4 对应 Full 学生 | 0.655348 | 86.025763 | 0.132445 | 0.058831 |
| count4 对应 Minimal 学生 | 0.656854 | 88.964175 | 0.135939 | 0.060444 |
| count4 对应 Velocity-only 学生 | 0.655127 | 88.886196 | 0.135477 | 0.060246 |
| count4 对应 Residual-only 学生 | 0.656394 | 86.033383 | 0.132724 | 0.058911 |

count4 相对 count8 提高多样性，但 FRC/FRD 有损失；原先 strict gate 失败后由用户明确接受此折中并蒸馏，不得改写为自动通过 strict gate。count4 的原报告“student not started”是当时状态，后续 `react2024_reference_count4_student_seed1` 已完成。

来源：`react2024_reference_count4_student_seed1/summary.json`、`react2024_student_loss_pruning_seed1/summary.json`。

## 关键组件能支持哪些结论

| 对照 | 数据支持 | 不支持的强化结论 |
|---|---|---|
| Mode prior 对 Uniform | Uniform FRDiv 下降约 51.8%；所有 571 context 均下降 | Mode prior 提升所有质量指标 |
| Minimal KD 对 Direct same GT | KD FRC 高 0.387677；Direct 的 FRD 和 FRDiv 反而更好 | 多样性只能靠蒸馏获得 |
| Residual 对 Minimal | 两数据集 FRD 明显改善；FRDiv 略降 | residual 是唯一多样性来源 |
| 删除 Velocity | 主要指标变化很小，可简化论文方法 | 数学上完全等价或所有指标严格不降 |
| 历史 Random 对 Max-FRDiv | 固定 GT 选择影响教师角色和多样性 | 已完成最终完整方法的 random-target 消融 |

2025 corrected Uniform − Mode 的 session-cluster bootstrap 95% 区间：FRC `[+0.528643,+0.801175]`；FRD `[-2.403709,+1.173301]`；FRDiv `[-0.113551,-0.085686]`。

2025 Direct − Minimal KD：FRC `[-0.446677,-0.295797]`；FRD `[-3.971230,-0.767244]`；FRDiv `[+0.001741,+0.009516]`。

这些区间来自每组固定 seed1 的 4000 次 session bootstrap，不是多 seed 训练置信区间。最新消融的匹配检查通过；具体初始化、缓存、source-order hash 和 source snapshots 均在原 run。

## 完整 VAL 随机十参考检查

预测和处理后 GT 固定；100 次完整 VAL571 重抽参考。以下 std 仅表示抽样变异，FRD 未重算，FRDiv 不变。

| 学生 | all-session FRC | paired 加 random9 FRC | uniform10 FRC |
|---|---:|---:|---:|
| Full | 1.444718 | 1.024583 ± 0.006425 | 1.019943 ± 0.007364 |
| Minimal | 1.454643 | 1.027358 ± 0.006522 | 1.022223 ± 0.007567 |
| Velocity only | 1.454373 | 1.026989 ± 0.006513 | 1.021893 ± 0.007578 |
| Residual only | 1.445507 | 1.025062 ± 0.006422 | 1.020433 ± 0.007353 |

结论：缩小 GT 池造成明显的 FRC 水平下降；完整数据集平均的重抽波动相对小。不能把“GT 池规模效应”与“随机 seed 方差”混为一谈。也不能声称完成随机参考 exact FRD 对照。

## 输入依赖诊断

冻结 2025 Residual-only 学生，不重新训练。六组完整 VAL571：

| 输入或控制 | FRC ↑ | exact FRD ↓ | FRDiv | 原始 paired mean CCC |
|---|---:|---:|---:|---:|
| 正常输入 | 1.445507 | 97.463281 | 0.170130 | 0.023024 |
| Audio 替换 TRAIN 均值 | 1.040990 | 107.261897 | 0.184895 | 0.008205 |
| Emotion 与 3DMM 替换 TRAIN 均值 | 1.463702 | 96.862408 | 0.176463 | 0.018947 |
| 三模态同步打乱 25-frame 时间块 | 1.347071 | 97.413746 | 0.170761 | 0.013755 |
| 固定十个 TRAIN 模板 | 0.635839 | 105.207555 | 0.298558 | 0.001041 |
| 按 speaker 特征最近邻检索 TRAIN 集合 | 0.790439 | 107.080023 | 0.276032 | 0.002138 |

另两组严格相同长度 750、同 session 可置换的 458-context cohort：

| 置换控制 | FRC ↑ | exact FRD ↓ | FRDiv | 原始 paired mean CCC |
|---|---:|---:|---:|---:|
| 该 cohort 正常 baseline | 1.346684 | 101.167203 | 0.162625 | 0.020274 |
| Same-session donor | 1.346905 | 101.159770 | 0.162625 | 0.010958 |
| Cross-session donor | 1.289366 | 102.561947 | 0.162625 | 0.011053 |

458 组不能与上面的 571 baseline 直接相减。置换使用同一批已缓存预测的双射，集合平均 FRDiv 不变是构造性质；不是模型崩塌或证明忽略输入。

判读：音频替换降低 FRC 约 28%，时间打乱也损伤 FRC；学生优于已测试模板与检索质量，支持输入依赖。但视觉均值替换并未降低 all-session 点估计，其 FRC/FRD 变化区间均跨零；原始 paired CCC 下降说明参考集合适配和原生配对同步不是同一问题。

原始 paired CCC 是补充指标，不是 official FRC，也不假定该反应是唯一正确答案。均值替换与打乱可能产生 OOD 输入；不能等同 retrained modality ablation，也不能宣称交互因果验证完毕。

来源：`react2025_input_dependence_seed1/summary.json`，完整分布、bootstrap、motion proxy 及 shard rows 均归档。该 run 总 wall 3248.34 秒。

## 历史路线索引与阶段决策

下表定位完整原报告，不抹去 pilot、失败和停止记录。每条的具体数值、参数、source hash 和当时协议以原文与原始 JSON 为准。

| 阶段 | 主要原报告 | 结论或地位 |
|---|---|---|
| 原 Mam 与 joint anchor | Mam docs 的技术归档、JOINT_ANCHOR、JOINT_3DMM | 代码来源与历史研究，不是当前 TCN 或学生结果 |
| M1 descriptor | M1_MAIN_RESULTS、M1_WEIGHT_SWEEP、M1_GRADIENT_DIAGNOSTICS | 大权重改善 coverage/diversity 但损 FRC |
| M1 长训练 | M1_MATCHED_LONG_RUN、M1_DETERMINISTIC_CONFIRMATION | 2500 steps / w0.10；后者修正随机 target processing |
| M2 GT ceiling | M2_GT_CEILING_SLOT_DIAGNOSTICS、DATASET_GT_ORACLE_CEILING | 区分 context-conditioned 与 session-balanced ceiling；未全求 exact |
| M2 adapter | M2_MODE_ADAPTER_V1 | 容量增加的负结果 |
| M2 routing | M2_GT_DERIVED_MODE_ROUTING、M2_ROUTE_WEIGHT_SWEEP | 行为身份增强，full VAL 仍非 strict win |
| M3 多 checkpoint Oracle | M3_ORACLE_SELECT10、M3_ORACLE_SELECT10_FULL | 80候选，GT-assisted；不是可部署方法或证明的组合最优 |
| 独立 fixed TCN 与 anchors | M3_FIXED_PAIR_ENSEMBLE、M3_ANCHOR_PAIRED_PLUS9_ENSEMBLE、M3_BEHAVIORAL_PAIRED_ANCHORS | 对齐政策、架构与预算不同，各自保留 |
| 早期学生与几何 KD | reports/student_relation_comparison、student_direction_comparison；fixed_teacher_distillation 子目录 | Basic、distance、direction、noise/spread 历史对照 |
| Certified candidate pool | reports/CERTIFIED_SELECT10_PROTOCOL 及 fixed_teacher_distillation 下 certified_select10 报告 | 下界、数值上界、gap；GT oracle 与安全 selector 分开 |
| Selector 与校准 | fixed_teacher_distillation 下 inference_safe、joint_selector、selector_diversity 报告 | 当时的 prototype teacher/student；不是当前最终算法组件 |
| 2024 早期蒸馏 | REACT2024_DISTILLATION、REACT2024_SPREAD_SCALE | 原生/转换口径必须逐行确认；部分历史路径为远端 |
| 固定 diverse GT | MAXFRDIV_TEACHER_TARGETS、MAXFRDIV_TEACHER_PILOT、REACT2024_T1_PURE_MAX_FRDIV | T0 random 与 T1 Max-FRDiv matched 600-step 对照 |
| Teacher quality gate | QUALITY_GATED_TARGET_FRONTIER、QUALITY_GATED_TEACHER_PILOT、QUALITY_GATED_TEACHER_FULL | 质量筛选与多样性折中，非最终模式先验 |
| 跨数据集差异 | CROSS_DATASET_MAXDIV_COMPATIBILITY | target representation / compatibility 诊断 |
| Graph ablation | GRAPH_ABLATION_T0 | Graph0 重训小幅损质量，post-hoc bypass FRD 大损 |
| 第一轮 T1 学生 | T1_STUDENT_DISTILLATION、T1_STUDENT_REFINEMENT、STUDENT_RECIPE_BANK | 保留 basic/residual/continuation/logit；后者不是 matched-diversity 全胜 |
| 教师更长训练 | TEACHER_CONVERGENCE | T1 1200/2400 未保持 600 的全部质量条件 |
| Target 处理和支持 | TEACHER_TARGET_RESEARCH、HYBRID_TEACHER | supported / processed / hybrid，多数 guard 未全通过 |
| Common VA | COMMON_VA、COMMON_VA_PILOT、LEARNED_COMMON_VA | 区分 GT oracle 和可部署修正，不选为最终主线 |
| Teacher 组合 | TEACHER_QUALITY_BAND | 替换/混合可达目标带，但预算及模型数不同，过于工程化未作主方法 |
| Conditional slot | CONDITIONAL_SLOT_PROBE | TRAIN cross-fit assignment 未胜简单对照，不推进训练 |
| 多参考教师 | LATENT_REFERENCE_TEACHERS；mode_reference_teacher_v2_seed1 | Uniform 提质量但压多样性；mode prior 恢复角色 |
| 最新蒸馏 | mode_reference_student_seed1；react2024_mode_reference_distillation_seed1 | 当前教师迁移到共享学生 |
| 2024 count4 | react2024_reference_count4_seed1 与对应 student | 用户接受质量/多样性折中后继续蒸馏 |
| 最新 loss deletion | 两数据集 student_loss_pruning；COMPONENT_ABLATION_RECORD_20260930 | 简化去 velocity；保留 residual 与 minimal 对照 |
| 最新机制对照 | key_component_ablations、input_dependence | corrected prior、DirectGT、输入置换与模板检索 |

原始文件的全名、来源、可点击副本都在 `ALL_REPORTS.md`；不得因名字相似合并两个 run。副本仍含当时的原始措辞，以下勘误优先。

## 有代表性的历史数值

这些不是最终路线的统一消融。早期 M1 / M2 / M3 使用 paired+9 确定性参考，而不是最终 all-session。

| 历史实验 | FRC | exact FRD | FRDiv |
|---|---:|---:|---:|
| M1 B1 full VAL571 | 0.798295 | 87.817214 | 0.006505 |
| M1 B3 w0.10 full VAL571 | 0.810111 | 82.480136 | 0.013611 |
| M2 route0.015 full VAL571 | 0.807115 | 82.375193 | 0.015051 |
| M3 strict GT oracle，80选10 | 0.864074 | 81.107519 | 0.033547 |
| M3 unconstrained heuristic | 0.829212 | 82.756349 | 0.036974 |

M1 在三个 seed 的 VAL64 和一个 seed 的 VAL571 通过其 strict 比较；M2 0.015 full VAL FRC 仍降 0.002996。M3 strict 每个 context 满足已验证质量约束，但数值仅代表当前算法找到的可行解。

下表为 **2025 all-session**，更长训练使用 fresh uninterrupted matched runs，不能与旧 frozen600 混作 bitwise 相同结果：

| Teacher | updates 每模型 | FRC | exact FRD | FRDiv |
|---|---:|---:|---:|---:|
| T0 random | 600 | 1.373312 | 84.928192 | 0.036411 |
| T1 Max-FRDiv | 600 | 1.072308 | 96.808993 | 0.205423 |
| T0 random | 1200 | 1.453088 | 84.920676 | 0.043459 |
| T1 Max-FRDiv | 1200 | 1.133364 | 97.645893 | 0.226178 |
| T0 random | 2400 | 1.524074 | 84.696737 | 0.062817 |
| T1 Max-FRDiv | 2400 | 1.024060 | 98.647957 | 0.238767 |

2024 原始 T0/T1 matched600：T0 `0.244023 / 83.538740 / 0.042493`；T1 `0.620408 / 87.107812 / 0.157259`。Max-FRDiv 在两数据集 FRC 方向不同，不代表数据集可直接比较难度。

原 T1 教师学生 bank：Basic `1.122329 / 100.690347 / 0.189725`；Residual `1.121615 / 96.617181 / 0.183968`；+5ep continuation `1.133005 / 96.705597 / 0.187008`；+5ep logit `1.132541 / 96.474092 / 0.186153`。这批教师不同于新 mode-reference，不能用旧学生权重冒充当前学生。

Graph 三路 matched T0 消融（2025 all-session，600updates）：

| 架构 | 每模型参数 | FRC | exact FRD | FRDiv |
|---|---:|---:|---:|---:|
| Graph2 linear | 3,720,129 | 1.373332 | 84.929128 | 0.036415 |
| Graph0 linear | 3,583,679 | 1.348490 | 85.689153 | 0.036882 |
| Graph0 MLP | 3,613,887 | 1.391284 | 87.306705 | 0.035777 |
| Graph2已训练权重推理bypass | 不重训 | 1.380108 | 142.738623 | 0.039572 |

MLP 实验确实完成，恢复 FRC 但损 FRD/FRDiv，因此当前冻结模型仍是 Graph2。不能因最终未采用而把该实验遗漏或写成未做。

后期 finite-pool certified oracle（旧 prototype 学生，all-session VAL571）另有：

| 候选池 | baseline FRDiv | 可行下界 | 数值上界 | selected FRC | selected FRD |
|---|---:|---:|---:|---:|---:|
| Raw30 | 0.060729 | 0.103495 | 0.106468 | 1.210950 | 85.360046 |
| Expanded60 | 0.060729 | 0.121067 | 0.129802 | 1.206121 | 85.481625 |

两组 baseline FRC=1.178483、FRD=85.709485；数值上界是 float64 allowance 的有限池诊断，不是区间算术形式证明。该实验不同于早期 M3 80候选，也不同于当前 mode-reference 学生；GT oracle 不算 inference-safe 部署结果。

## 勘误和不可混淆项

1. `M2_ROUTE_WEIGHT_SWEEP.md` 写“所有五点均 strict pass”与其表格矛盾：0.05 的 FRD=82.056537，大于 baseline82.047779；不是 strict pass。该文 0.02 的 VAL64 FRC delta 应为 +0.000436，而非表中负号。归档保留原文并在此更正，不静默修改历史。
2. 早期 `M3_ORACLE_SELECT10_FULL.md` 的“upper bound / maximum available”应理解为 oracle 可达性诊断。Greedy+swap 找到的是可行下界，不是证明的全局最优；后续 certified relaxation 文档的数值上界另列。
3. Dataset oracle 20个 session 仅17个证实 exact；session-balanced best-known0.237123 是整体最优的下界，不能称20个全 exact。旧 context-weighted farthest0.265977 又是不同长度/权重口径。
4. `epochs=50` 是教师名义 cosine schedule；当前600update不是50完整epoch。文件名 epoch0006 也不能替代实际 step metadata。
5. 原 uniform-v1 与 mode-v1 曾有 constant-prediction CCC 梯度处理修正；最终比较用 corrected mode和新 corrected uniform。旧值保留为历史，不覆盖最新值。
6. `TECHNICAL_ROADMAP.md` 是9月21日 candidate-pool 主线；现在是模式保持教师蒸馏。旧 roadmap 不再代表当前计划。
7. `CVPR_EVIDENCE_PLAN_20260930.json` 是当时待办清单，不表示列出的基线、人评、多 seed 已完成，也不因归档自动执行。
8. Count4 学生可接受不等于 strict gate 通过；图层未被最终模型删除；速度不是受控 benchmark；第五同步指标未在新主表补测。
9. 最新输入诊断重新聚合的 FRDiv0.1701303439 与原 student summary0.1701302826 存在约6e-8 float32聚合差异，不是新训练或性能提升。主学生表引用原 summary，输入诊断表引用该诊断的 matched baseline。
10. mode-reference trainer 的历史 `reference_prior` 描述字符串写死了 effective count8；数值参数可为4。2024应以 `reference_effective_count`、实际 prior audit 和 count4 run 为准，不能仅根据该描述字符串判断超参数。

## 数据划分记录与论文表述

`split_audit.json` 的内容哈希确认：本地2025 VAL/TEST的 facial、audio、coefficients 各1142文件相同。两者不能计作独立验证。TRAIN/VAL未发现完全相同的面部文件内容，但有101个共享文件名stem；这只标记 recording/identity overlap 待确认，不能凭 stem 宣判泄漏，也不能宣称已证明 subject-disjoint。

用户已接受当前数据条件并决定收敛实验。论文应诚实描述当前 local VAL 开发协议、GT 池规模、单 seed 和选择过程；“证据范围有限”不等于“所有现有结果无效”。

建议论文结构：任务矛盾 → 固定多样模式构造 → 模式保持参考边缘化 → 轻量共享学生 → 跨数据集主表 → 机制与loss删除 → 输入和协议敏感性 → 局限。无需加入已放弃的 sampler、noise、output blending 或 GT oracle 来增大方法图。
