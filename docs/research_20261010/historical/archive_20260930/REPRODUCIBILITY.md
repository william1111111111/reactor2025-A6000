> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 复现与归档文件定位

本次只整理已有证据，不重训、不重评主结果、不改默认超参数、不覆盖权重，也不自动推送远端。

## 环境和目录

| 内容 | 位置 |
|---|---|
| 工作仓库 | `$SOURCE_ROOT` |
| 持久项目 | `$PROJECT_ROOT` |
| Python | `$PROJECT_ROOT/.venv/bin/python` |
| Mam assets | `$PROJECT_ROOT/mam_reactor` |
| REACT2025 | `$PROJECT_ROOT/data` |
| 当前转换 REACT2024 | `$PROJECT_ROOT/runs/react2024_mam/v1/converted_t1_v2` |
| 持久 run 根 | `$RUN_ROOT` |
| 归档根 | `$ARCHIVE_ROOT` |

`environment_pip_freeze.txt` 为本次环境快照，不能保证所有旧 run 使用完全相同包版本。`git_head.txt`、`git_status.txt`、`working_tree.patch` 和复制的代码共同标识当前工作树。最近实验 Git base 为 `c421f509f8cb7997e9341592cae2eb1e721b0e8e`，但有大量本地修改；复现必须优先看各 run 自带 code/source snapshot 与 SHA。

普通运行环境示例，仅说明依赖，不在归档时执行训练：

```bash
export MAM_REACTOR_ROOT=$PROJECT_ROOT/mam_reactor
export PYTHONPATH=$SOURCE_ROOT/mam_reactor:$SOURCE_ROOT
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
```

历史允许使用物理 GPU1/5/6。配置中 `device=cuda:0` 可能是 `CUDA_VISIBLE_DEVICES` 重映射后的局部编号，不自动表示用物理 GPU0；以 launcher/experiment 的 gpus 信息为准。

## 当前权重注册

以下路径均相对于持久 run 根，完整权重列表和重新计算 SHA 在 `weights_and_caches.csv`。

| 角色 | 相对路径 | 已记录 SHA256 |
|---|---|---|
| 2025 简化学生 | `react2025_student_loss_pruning_seed1/residual/student.pth` | `84df1ea5cd51196747fa662421653ff5c8789324221a52529053a6612e1935b4` |
| 2024 简化学生 | `react2024_student_loss_pruning_seed1/residual/student.pth` | `0c41afe63a4e56ad4ec0f7722870119ee8bf209616e0b23157199f53ddd32eb5` |
| DirectGT control | `react2025_key_component_ablations_seed1/direct_student/student.pth` | `33c040fea67ec74a45f421b95d23e6e0486aeffcae16d0b4ec5d2c734bc549e5` |

2025 teacher q0 在 `teacher_convergence_2400_seed1/arms/T1/rank_00/conditional-step000600-seed1.pth`；q1..q9 在 `mode_reference_teacher_v2_seed1/rank_01` 到 `rank_09/conditional-step000600-seed1.pth`。完整有序 bank 与十个 hash 见 `mode_reference_student_seed1/experiment.json` 和其 TRAIN cache manifest。

注意：学生初始化 checkpoint 为 `maxfrdiv_teacher_pilot_600_v3/arms/T1/rank_00/conditional-epoch0006-seed1.pth`，文件 hash `ffdc6dabb58a96c521eeacc93c74a81298a93b1e14d1cb7060e06140b235bea4`；不要将 teacher bank 的 fresh-q0 文件与学生 initializer 文件混淆。

2024 bank 以 `react2024_reference_count4_student_seed1/experiment.json`、`train_cache/manifest.json` 的有序 teacher_paths/hash 为准。paired initializer 位于 `react2024_mode_reference_distillation_seed1/frozen_T1/rank_00/conditional-epoch0004-seed1.pth`。

主表、配置、训练日志、完整 eval 与 hash 应成套使用。只拿文件名“best”或最后一个 epoch，不能保证重现对应主表。

## 关键记录对应关系

| 问题 | Run 或文档 | 优先检查的文件 |
|---|---|---|
| 2025 最新学生loss消融 | react2025_student_loss_pruning_seed1 | summary / experiment / frozen_full_provenance / 各arm config |
| 2024 最新学生loss消融 | react2024_student_loss_pruning_seed1 | 同上 |
| corrected prior 与直接GT | react2025_key_component_ablations_seed1 | summary / experiment / 各项 matching audit |
| 最新输入依赖 | react2025_input_dependence_seed1 | summary / frozen_plan / split_audit / shard rows |
| 2025 teacher→student | mode_reference_student_seed1 | teacher_upgrade_comparison / experiment / cache manifest |
| 2024 teacher→student | react2024_mode_reference_distillation_seed1 | summary / experiment / old/new cache |
| 2024 count4接受版本 | react2024_reference_count4_student_seed1 | summary / experiment / teacher hashes |
| 原T0/T1与长训练 | maxfrdiv_teacher_pilot_600_v3；teacher_convergence_2400_seed1 | manifests / configs / snapshot steps / 全VAL |
| 历史学生全部recipe | student_recipe_bank_20260924；fixed_teacher_distillation | registry / REPORT / provenance |
| 路线历史文档 | repo coreset/results 与 reports | ALL_REPORTS.md 可点击副本 |

学生简化版可按原 TRAIN cache 和冻结校准重训；为了不误触新实验，此处不提供直接执行整队列的命令。源码入口及全部实际参数已经收录，重跑时应新建 run-dir，保留原结果。

## 主要程序入口

| 阶段 | 文件 |
|---|---|
| 固定多样目标 | prepare_maxfrdiv_teacher_manifests.py，maxfrdiv_teacher_manifest.py |
| 教师普通训练 | mam_reactor/regnn/train_conditional_regnn.py |
| Uniform 多参考 | train_latent_reference_teacher.py |
| Mode-reference | train_mode_reference_teacher.py |
| 当前教师学生流水线 | run_mode_reference_student.py，run_react2024_mode_distillation.py |
| TRAIN teacher cache | teacher_cache.py |
| 共享学生训练 | train_expert_student.py |
| 残差等KD目标 | relation_distillation.py |
| 学生loss删除 | run_student_loss_pruning.py |
| 关键组件消融 | run_key_component_ablations.py，build_matched_real_gt_cache.py |
| 完整快速评估 | eval_expert_student_fast.py，fast_frc.py，oracle_select.SharedQualityPool |
| 随机参考FRC | eval_random_reference_frc.py |
| 输入依赖诊断 | input_dependency_audit.py |
| 研究归档 | archive_research.py |

## 归档格式与验证

`file_manifest.jsonl` 每行登记一个原始文件。`storage=copied` 表示文本已复制到 evidence 并逐文件核对 SHA；`in_place_not_copied` 表示权重或大张量仍在原地。`sha256=null` 表示未计算，不能当成校验通过。

`metric_index.csv` 保留 JSON pointer；自动提取器不推测历史 reference_count、数据集、完成状态或模型身份。某些行是 delta / intermediate / protocol-specific 值，不能只按最大 FRC 排序后写成主结果。非字典数组（逐帧、逐样本、逐rank）保存在原始 evidence，不丢弃。

`missing_or_external_weight_references.json` 是旧记录引用但本机不可定位的权重清单；远端历史路径并非自动判定文件丢失。实际扫描错误另外列入 `collection_errors.json`；未恢复的缺口不能编造结果补上。

本次该清单含19个旧外部路径及1个失效的仓库内 descriptor cache 路径。后者原为 `$SOURCE_ROOT/coreset_reactor/cache/train_descriptors_v2.pt`；持久位置 `$PROJECT_ROOT/coreset_reactor/cache/train_descriptors_v2.pt` 实际存在且已入缓存索引。这是可检查的定位候选，不据同名自动声称其内容与旧引用版本相同，也不重写历史配置。

归档不加载 `.pth/.pt` pickle，不读取原始视频，不访问 SSH key，不联网，不使用 GT 重新选方案。权重 hash 是流式读，不改变内容。原日志、状态和历史负结果保持原样。

可复核方式：用任一 SHA256 工具验证清单对应路径；对主表查原 summary 的 arm；对训练超参查每个 checkpoint 的 config / provenance；对数据语义查 loader 快照；对结果时间点查原 mtime 和 status，而不是文档里单独的“完成”字样。

完整归档根的 `ARCHIVE_SHA256SUMS.txt`（封存阶段生成）覆盖归档内部文件，`VERIFICATION.json` 记录主表、权重与副本检查。它们校验归档文件一致性，不保证所有历史科学结论正确。

## 发布与后续边界

本次没有 git push；归档中包含私有元数据，不能整体提交公共仓库。若以后要发 GitHub，另生成只含方法、聚合表、匿名配置与依赖的发布包，并由用户确认。

当前不再自动扩实验。论文写作可从主方法和已完成消融开始；独立TEST、人评、最终路线多seed、外部强基线若未做，应如实保持“未验证”，不为填完整表而补造结论。
