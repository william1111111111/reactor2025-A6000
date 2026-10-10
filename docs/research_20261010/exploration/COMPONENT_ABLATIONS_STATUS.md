> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 学生结构与蒸馏组合消融执行状态

2026年9月30日，两项消融已实施并由独立服务启动正式训练，尚无完整新结果。
按用户要求不增加训练随机种子。旧实验、原项目和失败记录均不覆盖。

## 已完成的检查

- 57项测试通过，包括16项本轮新增测试；主机与服务内各完整通过一次。
- 3320个TRAIN source/crop/view、教师缓存分片SHA、教师概率语义全部核验。
- 基础KD、仅集合匹配、仅动态项各精确重放两步；日志标量、模型、优化器、
  scheduler及CPU/CUDA/NumPy/Python/DataLoader RNG与历史checkpoint一致。
- 两个新增方法各完成两步真实GPU smoke、模型审计、三条VAL的完整推理与all-session精确评分。
- 新专家模型加载路径在三条VAL上与旧构造器输出逐元素一致。
- 结构对照共同预训练权重一致，初始化RNG一致；完整初始函数不同，不宣称完全相同初始化。

## 正式实验

| 新增方法 | GPU | 训练预算 | 对照 |
|---|---:|---:|---|
| 集合匹配加动态约束 | 5 | 8300步 | 复用基础、仅匹配、仅动态，组成四格实验 |
| 共享骨干加十个独立线性输出头 | 1 | 8300步 | 复用专家ID调制的基础KD |

原专家结构3,803,201参数，十头结构3,777,954参数，少25,247参数，约0.66%。
十头对照保留原关系图模块；没有新增随机种子、没有改成剪枝、没有同时更换教师或损失。
组合损失两项共用同一个整段匹配后的教师目标，动态系数固定1.0。

服务为 `react-component-ablations-20260930.service`，目录为 `runs/component_ablations_v1/`。
GPU6及其他任务未动。正式训练共16,600个新优化步；随后自动完成VAL571的all-session
精确评分、连续残差诊断、四格交互分析及GPU5推理微基准。
当前文字是启动时快照，不代表实时步数或已完成。

## 查询与恢复边界

在本目录执行以下只读查询，需要主机用户服务总线权限：

```bash
$LOCAL_PATH/python component_ablations_status.py
```

以服务存活、5秒心跳、每100步的PROGRESS和最终COMPLETE标记共同判断进度。
每完整epoch原子保存模型、优化器、scheduler和RNG。单作业2小时、服务6小时上限，
失败停止，不自动重试。不能重复启动，也不能向旧目录盲目resume。
中断后先核验最后完整checkpoint，再在独立恢复目录制定恢复步骤。

最终完成需要 `eval/analysis.json`、`eval/RESULTS.md`、`eval/BENCHMARK.json`
以及服务stage为complete。正常结束后的inactive不表示失败。

所有指标与负结果均保留。session bootstrap不是新增训练种子，也不覆盖训练随机性。
结构只控制共同预训练权重与数据预算，参数量不完全相等；微基准不等于端到端速度，
PyTorch profiler的FLOPs只是支持算子的部分计数。动态能量不是有效反应覆盖率。

冻结配置和源码副本见 `runs/component_ablations_v1/PREPARED.json` 与 `source_snapshot/`。
入口：[冻结协议](COMPONENT_ABLATIONS_PROTOCOL.md)、[消融总索引](ABLATION_INDEX.md)。
