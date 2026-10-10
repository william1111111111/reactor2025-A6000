> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 消融保留总索引

2026-09-28，按用户要求保留所有已有消融，包括负结果、中断记录和 smoke。
不删除、不覆盖历史 checkpoint、缓存、逐样本预测、日志或评测；新方法另建目录。

2026-09-30更新：三种新蒸馏已完成。两项新组件消融在独立 `runs/component_ablations_v1/`
启动，见 `COMPONENT_ABLATIONS_STATUS.md`；不增加训练随机种子。
归档清单是此前时间点的快照，不自动包含新实验；
原元数据副本不改写，导航文档可以追加新状态。

## 保存范围与完整性

- 当前探索目录的全部 `runs/` 原位保留。
- 原项目 `$RUN_ROOT/` 全部原位保留，
  不只保留表格中表现较好的方法；本轮没有修改原项目。
- `preserve_ablations.py create` 创建一次性 `preserved/ablations_20260928_v1/`：
  全量文件路径/大小/时间清单；小型报告、源码、配置及日志的内容寻址副本；
  当前 runs 的非缓存文件、原 reports 的模型权重和被复制文本的 SHA256。
- 原项目的大型 teacher cache/预测文件仅记录路径、大小、mtime，未承诺逐字节复核。
  编译缓存明确排除。数据集与原始视频不在此次快照范围。
- 这是**原位保留 + 元数据快照**，不是模型/数据的异地完整备份，也不抵御原磁盘损坏。
  归档成功以 `COMPLETE.json` 为准，`verify --full` 重算所有已登记的内容哈希。
- 不修改文件权限、不移动目录、不通过 hardlink 假装独立备份、不提交到远程仓库。

## 最近三轮与恢复链

| 记录 | 内容 | 状态/解释 | 入口 |
|---|---|---|---|
| 首轮冻结模型诊断 | 2024/2025 输入反事实及参考集合诊断 | 保留，更正后的协议解释优先 | `REPORT.md`、`runs/final_analysis_v1/summary.json` |
| 匹配教师监督 | fixed / uniform / mode / shuffled，36 个 specialist | 训练及完整 VAL571 已完成；没有证明 mode 更优的质量—多样性边界 | `SECOND_ROUND_REPORT.md`、`runs/matched_supervision_v1/FINAL_AUDIT.json` |
| 原学生对照 | direct_fixed / direct_mode 的原执行及四组 smoke | 原正式进程中断，两个可恢复 checkpoint 均为 1660 步；旧 status 不代表存活 | `runs/student_method_v1/` |
| 恢复后的完整学生对照 | 两组 direct + basic KD + residual KD | 全部 8300 步，完整 VAL571/all-session 精确评分与统计完成 | `STUDENT_METHOD_RESULTS.md`、`runs/student_method_recovery_v1/eval/analysis.json` |
| 不同蒸馏方式 | typed、assignment、dynamic 与复用的基础和残差KD | 三组均8300步与完整VAL571完成，含连续残差诊断 | `runs/distillation_methods_v1/eval/RESULTS.md` |
| 学生结构与蒸馏组合 | 十输出头基础KD、assignment加dynamic | 57测试及精确重放通过，GPU1/5正式训练已启动，尚无新完整结果 | `COMPONENT_ABLATIONS_STATUS.md` |
| 2024 native 协议审计 | neighbor 与索引行对应 | 尚未认证，不能作为正式跨数据集结果 | `runs/native2024_protocol_v1/result.json` |

## 原项目历史蒸馏/消融（路径相对原 reports）

| 路径 | 已有方法 | 必须保留的比较边界 |
|---|---|---|
| `t1_student_distillation_seed1/` | basic、distance、contrast/residual、AU decision | T1 教师、8300 步；不能与新 mode 教师混表归因于损失 |
| `t1_student_refinement_seed1/` | 继续训练 control、AU decision、AU logit | 额外 4150 步；必须对比同预算 control，不只对比 epoch10 父模型 |
| `student_recipe_bank_20260924/` | 历史配方索引及权重哈希 | 推荐列表不是删除筛选器；旧“官方 sampled-10”措辞已被新协议说明纠正 |
| `mode_reference_student_seed1/` | mode 教师下 basic、contrast/residual | 旧 DataLoader 日程；与新四组对照不同实验批次 |
| `react2024_student_loss_pruning_seed1/` | full、minimal、velocity、residual | 是删减训练损失，不是通道/结构剪枝；2024 native 协议待认证 |
| `react2025_student_loss_pruning_seed1/` | 速度项与残差项完整四格对照 | 已有REACT2025 VAL571/all-session结果；旧日程，不与新蒸馏批次混作完全匹配实验 |
| `react2024_mode_reference_distillation_seed1/`、`react2024_reference_count4_student_seed1/` | 2024 的教师/学生及参考数实验 | 保留历史结果与限制，不冒充认证的 native 结果 |
| 其余所有 reports 子目录 | 教师选择、教师升级、图结构、M1/M2 等历史探索 | 文件全部纳入清单；此处不代表重新认证所有旧结论 |

## 以后继续工作的规则

1. 评测以 `PROTOCOL_AUTHORITY_CORRECTION.md` 为准：2025 all-session 集合有论文依据，
   十条预测不等于十条 GT；不更换成 sampled-10 主评测。
2. 完成标记、模型与逐样本结果优先于旧手工 status 文案。
3. 新方法目录不可复用旧目录；失败/无改善结果也保留。
4. 学生主干、十候选及 T1→教师→学生路线不因归档而改变；不新开剪枝路线。
5. 调研记录见 `DISTILLATION_REVIEW_20260928.md`；后续获授权执行的状态另见
   `DISTILLATION_METHODS_STATUS.md`，不要把方案文档当成最终训练结果。
